import torch
import gc


def kill_vllm_process(llm):
    del llm

    gc.collect()
    torch.cuda.empty_cache()


def get_assistant_turn_token_boundaries(conversation, tokenizer):
    assistant_indices = [
        i for i, m in enumerate(conversation) if m["role"] == "assistant"
    ]
    if not assistant_indices:
        raise ValueError("No assistant messages found.")

    # Determine which assistant message to score
    target_idx = assistant_indices[-1]

    # Get two tokenized versions:
    # 1. Up to and including this assistant turn
    with_assistant = tokenizer.apply_chat_template(
        conversation[: target_idx + 1], tokenize=True, return_tensors="pt"
    )

    # 2. Up to just before this assistant turn
    without_assistant = tokenizer.apply_chat_template(
        conversation[:target_idx], tokenize=True, return_tensors="pt"
    )

    # The difference in lengths tells us which tokens correspond to the assistant message
    start = without_assistant.shape[1]
    end = with_assistant.shape[1]

    return start, end
