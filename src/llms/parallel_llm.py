from typing import Generator, Iterable, cast

from .._cachable import _ParallelCachable
from .llm import DEFAULT_BATCH_SIZE, LLM


class ParallelLLM(_ParallelCachable, LLM):
    def __init__(self, *llms: LLM):
        super().__init__(*llms, cls=LLM)
        self.llms = cast(list[LLM], self.cachables)

    def count_tokens(self, value: str) -> int:
        return self.llms[0].count_tokens(value=value)

    def get_max_context_length(self, max_new_tokens: int) -> int:
        return self.llms[0].get_max_context_length(max_new_tokens=max_new_tokens)

    def format_prompt(  # noqa: C901
        self,
        max_new_tokens: None | int = None,
        beg_instruction: None | str = None,
        in_context_examples: None | list[str] = None,
        end_instruction: None | str = None,
        sep="\n",
        min_in_context_examples: None | int = None,
        max_in_context_examples: None | int = None,
    ) -> str:
        return self.llms[0].format_prompt(
            max_new_tokens=max_new_tokens,
            beg_instruction=beg_instruction,
            in_context_examples=in_context_examples,
            end_instruction=end_instruction,
            sep=sep,
            min_in_context_examples=min_in_context_examples,
            max_in_context_examples=max_in_context_examples,
        )

    def run(
        self, prompts: Iterable[str], *args, **kwargs
    ) -> Generator[str | list[str], None, None] | list[str | list[str]]:
        kwargs["batch_size"] = kwargs.pop("batch_size", DEFAULT_BATCH_SIZE)
        results_generator = self._run_in_parallel(prompts, *args, **kwargs)
        if not kwargs.get("return_generator", False):
            return list(results_generator)
        else:
            return results_generator

    def unload_model(self):
        for llm in self.llms:
            llm.unload_model()


__all__ = ["ParallelLLM"]
