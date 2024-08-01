from typing import Callable

import pytest

from ....llms import LLM


@pytest.fixture
def mock_llm(
    allowed_kwargs=frozenset(
        {
            "inputs",
            "batch_size",
            "max_new_tokens",
            "temperature",
            "top_p",
            "n",
            "stop",
            "repetition_penalty",
            "logit_bias",
            "seed",
            "max_length_func",
            "cached_tokenizer",
        }
    ),
) -> Callable[..., LLM]:
    def _mock_llm(llm: LLM, responses: dict[str, str]) -> LLM:
        def _run_batch_mocked(**kwargs):
            for kwarg in kwargs:
                assert kwarg in allowed_kwargs, f"LLM got unexpected keyword: {kwarg}"
            return [responses[prompt] for prompt in kwargs["inputs"]]

        llm._run_batch = _run_batch_mocked  # type: ignore[attr-defined]

        return llm

    return _mock_llm
