from typing import Generator, Iterable, cast

from .._cachable import _ParallelCachable
from .llm import DEFAULT_BATCH_SIZE, LLM


class ParallelLLM(_ParallelCachable, LLM):
    def __init__(self, *llms: LLM):
        """
        Creates a LLM that will run multiple LLMs in parallel. See
        :doc:`running models in parallel
        <./pages/advanced_usage/parallelization/running_models_on_multiple_gpus>`
        for more details.

        Args:
            *llms: The LLMs to run in parallel.
        """
        super().__init__(*llms, cls=LLM)
        self.llms = cast(list[LLM], self.cachables)

    def count_tokens(self, value: str) -> int:
        """Counts the number of tokens in a string.

        Args:
            value: The string to count tokens for.

        Returns:
            The number of tokens in the string.
        """
        pass
        return self.llms[0].count_tokens(value=value)

    def get_max_context_length(self, max_new_tokens: int) -> int:
        """Gets the maximum context length for the model. When ``max_new_tokens`` is
        greater than 0, the maximum number of tokens that can be used for the prompt
        context is returned.

        Args:
            max_new_tokens: The maximum number of tokens that can be generated.

        Returns:
            The maximum context length.
        """
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
