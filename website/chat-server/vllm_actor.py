import ray
from typing import List, Dict, Any, Optional
from vllm.sampling_params import SamplingParams
import asyncio
import os
import sys
import asyncio
import time


@ray.remote
class GPUInitScheduler:
    def __init__(self):
        self.locked = {}
        self.MIN_COOLDOWN = 30

        for i in range(8):
            self.locked[i] = False


    async def acquire_lock(self, gpu_id):
        while self.locked[gpu_id]:
            await asyncio.sleep(1)

        self.locked[gpu_id] = True

    async def release_lock(self, gpu_id):
        self.locked[gpu_id] = False
                
                


@ray.remote(num_gpus=0.5)
class VllmActor:
    def __init__(
        self,
        model_path: str,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.45,
        max_model_len: Optional[int] = 8192,
        dtype: str = "auto",
        gpu_init_scheduler = None,
        **kwargs
    ):
        """
        Initialize the VLLM Actor with an AsyncLLMEngine.

        Args:
            model_path: Path to the model
            tensor_parallel_size: Number of GPUs for tensor parallelism
            gpu_memory_utilization: GPU memory utilization (0-1)
            max_model_len: Maximum model context length
            dtype: Data type for model weights
            **kwargs: Additional engine arguments
        """
        from transformers import AutoTokenizer

        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.engine.async_llm_engine import AsyncLLMEngine
        from vllm.entrypoints.launcher import serve_http
        from vllm.entrypoints.utils import with_cancellation
        from vllm.logger import init_logger
        from vllm.usage.usage_lib import UsageContext
        from vllm.utils import FlexibleArgumentParser, random_uuid, set_ulimit
        from vllm.version import __version__ as VLLM_VERSION

        os.environ['VLLM_USE_V1'] = '1'

        self.model_path = model_path
        self.engine = None

        # Initialize tokenizer for chat template formatting
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        ray.get(gpu_init_scheduler.acquire_lock.remote(ray.get_gpu_ids()[0]))

        # Prepare engine arguments
        engine_args = AsyncEngineArgs(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=True,
            dtype=dtype,
            max_model_len=6144,
            max_num_seqs=4,
            max_num_batched_tokens=6144,
            enforce_eager=True,
            **kwargs
        )

        # Initialize the async engine
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)

        ray.get(gpu_init_scheduler.release_lock.remote(ray.get_gpu_ids()[0]))

    async def chat(self, conversation: List[Dict[str, str]], extra_params: Dict = {}):
        """
        Process a chat conversation and return the response.

        Args:
            conversation: List of message dictionaries with 'role' and 'content'
            extra_params: Additional sampling parameters

        Returns:
            Generated response text
        """
        from vllm.utils import random_uuid
        # Use tokenizer's apply_chat_template to format the conversation
        prompt = self.tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True
        )

        # Set up sampling parameters
        sampling_params = SamplingParams(
            temperature=extra_params.get('temperature', 0.5),
            max_tokens=extra_params.get('max_tokens', 4096),
        )

        # Generate a unique request ID
        request_id = random_uuid()

        # Generate the response
        results_generator = self.engine.generate(prompt, sampling_params, request_id)

        # Collect the final result
        final_output = None
        async for request_output in results_generator:
            final_output = request_output

        # Extract and return the generated text
        if final_output and final_output.outputs:
            return final_output.outputs[0].text

        await asyncio.sleep(0.001)

        raise Exception("No final output from model!")

    async def shutdown(self):
        """
        Shutdown the engine and cleanup resources.
        """
        sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

        from utils.vllm import kill_vllm_process

        if self.engine:
            # Note: AsyncLLMEngine doesn't have a direct shutdown method
            # Setting to None allows garbage collection
            self.engine = None
            # If there's a specific kill function needed:
            kill_vllm_process(self.engine)

            await asyncio.sleep(0.1)