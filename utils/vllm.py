import torch
import gc


def kill_vllm_process(llm):
    del llm

    gc.collect()
    torch.cuda.empty_cache()
