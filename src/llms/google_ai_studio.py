from functools import cached_property
from typing import Callable

from ._litellm import LiteLLM
from .llm import DEFAULT_BATCH_SIZE


class GoogleAIStudio(LiteLLM):
    def __init__(
        self,
        model_name: str,
        api_key: None | str = None,
        retry_on_fail: bool = True,
        cache_folder_path: None | str = None,
        **kwargs,
    ):
        super().__init__(
            model_name=model_name,
            api_key=api_key,
            retry_on_fail=retry_on_fail,
            cache_folder_path=cache_folder_path,
            **kwargs,
        )
        self._model_name_prefix = "gemini/"

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
    ) -> list[str] | list[list[str]]:  # pragma: no cover
        assert (
            repetition_penalty is None
        ), f"`repetition_penalty` is not supported for {type(self).__name__}"
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
        return "https://arxiv.org/abs/2312.11805"

    @cached_property
    def license(self) -> None | str:
        return "https://ai.google.dev/gemini-api/terms"

    @cached_property
    def citation(self) -> None | list[str]:
        citations = []
        citations.append(
            """@article{anil2023gemini,
  title={Gemini: A family of highly capable multimodal models},
  author={Anil, Rohan and Borgeaud, Sebastian and Wu, Yonghui and Alayrac, Jean-Baptiste and Yu, Jiahui and Soricut, Radu and Schalkwyk, Johan and Dai, Andrew M and Hauth, Anja and Millican, Katie and others},
  journal={arXiv preprint arXiv:2312.11805},
  volume={1},
  year={2023}
}""".strip()
        )
        return citations


__all__ = ["GoogleAIStudio"]
