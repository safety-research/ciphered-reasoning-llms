from transformers import AutoTokenizer

def get_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if model_name == "unsloth/Llama-3.1-8B":
        chat_tokenizer = AutoTokenizer.from_pretrained("unsloth/Meta-Llama-3.1-8B-Instruct")
        tokenizer.chat_template = chat_tokenizer.chat_template
        tokenizer.eos_token = chat_tokenizer.eos_token
        tokenizer.bos_token = chat_tokenizer.bos_token

    return tokenizer