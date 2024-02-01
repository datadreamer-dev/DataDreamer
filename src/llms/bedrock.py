from functools import cached_property
from typing import Callable

from ._litellm import LiteLLM
from .llm import DEFAULT_BATCH_SIZE


class Bedrock(LiteLLM):
    def __init__(
        self,
        model_name: str,
        aws_access_key_id: None | str = None,
        aws_secret_access_key: None | str = None,
        aws_region_name: None | str = None,
        retry_on_fail: bool = True,
        cache_folder_path: None | str = None,
        **kwargs,
    ):
        super().__init__(
            model_name=model_name,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            aws_region_name=aws_region_name,
            retry_on_fail=retry_on_fail,
            cache_folder_path=cache_folder_path,
            **kwargs,
        )
        self._model_name_prefix = ""

    def _run_batch(
        self,
        max_length_func: Callable[[list[str]], int],
        inputs: list[str],
        max_new_tokens: None | int = None,
        temperature: float = 1.0,
        top_p: float = 0.0,
        n: int = 1,
        stop: None | str | list[str] = None,
        repetition_penalty: None | float = None,
        logit_bias: None | dict[int, float] = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        seed: None | int = None,
        **kwargs,
    ) -> list[str] | list[list[str]]:
        assert (
            repetition_penalty is None
        ), f"`repetition_penalty` is not supported for {type(self).__name__}"
        assert n == 1, f"Only `n` = 1 is supported for {type(self).__name__}"
        return super()._run_batch(
            max_length_func=max_length_func,
            inputs=inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            n=n,
            stop=stop,
            repetition_penalty=repetition_penalty,
            logit_bias=logit_bias,
            batch_size=batch_size,
            seed=seed,
            **kwargs,
        )

    @cached_property
    def model_card(self) -> None | str:
        return "https://aws.amazon.com/bedrock/"

    @cached_property
    def license(self) -> None | str:
        return "https://aws.amazon.com/terms/"


__all__ = ["Bedrock"]
