from functools import cache

TOGETHER_TOKENIZERS = {
    "togethercomputer/Pythia-Chat-Base-7B-v0.16": "togethercomputer/Pythia-Chat-Base-7B",
    "togethercomputer/Qwen-7B-Chat": "Qwen/Qwen-7B-Chat",
    "togethercomputer/Koala-13B": "meta-llama/Llama-2-13b-hf",
    "togethercomputer/llama-2-7b-chat": "meta-llama/Llama-2-7b-chat-hf",
    "togethercomputer/llama-2-13b-chat": "meta-llama/Llama-2-13b-chat-hf",
    "togethercomputer/llama-2-70b-chat": "meta-llama/Llama-2-70b-chat-hf",
    "togethercomputer/CodeLlama-7b-Instruct": "codellama/CodeLlama-7b-Instruct-hf",
    "togethercomputer/CodeLlama-13b-Instruct": "codellama/CodeLlama-13b-Instruct-hf",
    "togethercomputer/CodeLlama-34b-Instruct": "codellama/CodeLlama-34b-Instruct-hf",
    "togethercomputer/mpt-7b-chat": "mosaicml/mpt-7b-chat",
    "togethercomputer/mpt-30b-chat": "mosaicml/mpt-30b-chat",
    "togethercomputer/alpaca-7b": "huggyllama/llama-7b",
    "togethercomputer/falcon-7b-instruct": "tiiuae/falcon-7b-instruct",
    "togethercomputer/falcon-40b-instruct": "tiiuae/falcon-40b-instruct",
    "togethercomputer/guanaco-7b": "huggyllama/llama-7b",
    "togethercomputer/guanaco-13b": "huggyllama/llama-13b",
    "togethercomputer/guanaco-33b": "huggyllama/llama-30b",
    "togethercomputer/guanaco-65b": "huggyllama/llama-65b",
    "togethercomputer/Qwen-7B": "Qwen/Qwen-7B",
    "togethercomputer/llama-2-7b": "meta-llama/Llama-2-7b-hf",
    "togethercomputer/llama-2-13b": "meta-llama/Llama-2-13b-hf",
    "togethercomputer/llama-2-70b": "meta-llama/Llama-2-70b-hf",
    "togethercomputer/mpt-30b": "mosaicml/mpt-30b",
    "togethercomputer/mpt-30b-instruct": "mosaicml/mpt-30b-instruct",
    "togethercomputer/falcon-7b": "tiiuae/falcon-7b",
    "togethercomputer/falcon-40b": "tiiuae/falcon-40b",
    "togethercomputer/codegen2-7B": "Salesforce/codegen2-7B",
    "togethercomputer/codegen2-16B": "Salesforce/codegen2-16B",
    "togethercomputer/CodeLlama-7b": "codellama/CodeLlama-7b-hf",
    "togethercomputer/CodeLlama-13b": "codellama/CodeLlama-13b-hf",
    "togethercomputer/CodeLlama-34b": "codellama/CodeLlama-34b-hf",
    "togethercomputer/CodeLlama-7b-Python": "codellama/CodeLlama-7b-Python-hf",
    "togethercomputer/CodeLlama-13b-Python": "codellama/CodeLlama-13b-Python-hf",
    "togethercomputer/CodeLlama-34b-Python": "codellama/CodeLlama-34b-Python-hf",
}


@cache
def _model_name_to_tokenizer_model_name(model_name: str) -> str:  # pragma: no cover
    model_name_lower = model_name.lower()
    if all(fragment in model_name_lower for fragment in ["llama-", "-2-", "-chat"]):
        return "meta-llama/Llama-2-7b-chat-hf"
    return "gpt2"


__all__ = ["TOGETHER_TOKENIZERS"]
