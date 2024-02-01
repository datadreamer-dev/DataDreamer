from typing import Callable

import pytest

from ....llms import LLM


@pytest.fixture
def mock_llm() -> Callable[..., LLM]:
    def _mock_llm(llm: LLM, responses: dict[str, str]) -> LLM:
        def _run_batch_mocked(**kwargs):
            return [responses[prompt] for prompt in kwargs["inputs"]]

        llm._run_batch = _run_batch_mocked  # type: ignore[attr-defined]

        return llm

    return _mock_llm
