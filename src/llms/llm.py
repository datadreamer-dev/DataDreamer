from abc import abstractmethod
from functools import cached_property
from itertools import chain
from typing import Any, Callable, Generator, Iterable, cast
from uuid import uuid4

from .._cachable import _Cachable
from .._cachable._cachable import _StrWithSeed

DEFAULT_BATCH_SIZE = 10


class _NoRetryException(Exception):
    pass


def _check_max_new_tokens_possible(
    self: "LLM",
    max_length_func: Callable[[list[str]], int],
    prompts: list[str],
    max_new_tokens: None | int = None,
) -> int:
    # Get max prompt length
    max_prompt_length = max_length_func(prompts)

    # Check max_new_tokens
    max_context_length = self.get_max_context_length(max_new_tokens=0)
    max_output_length = self._get_max_output_length()
    max_new_tokens_possible = self.get_max_context_length(
        max_new_tokens=max_prompt_length
    )
    if max_new_tokens_possible > 0 and max_new_tokens is None:
        max_new_tokens = min(
            max_new_tokens_possible, max_output_length or max_new_tokens_possible
        )
    elif max_output_length is not None and (
        max_new_tokens is not None and max_new_tokens > max_output_length
    ):
        raise ValueError(
            "The requested max_new_tokens exceeds the maximum output length of the"
            " model."
        )
    elif (
        max_new_tokens_possible <= 0
        or (max_new_tokens is not None and max_new_tokens_possible < max_new_tokens)
        or max_prompt_length > max_context_length
    ):
        raise ValueError(
            "The length of your prompts and requested max_new_tokens exceeds the"
            " context length of the model."
        )
    assert isinstance(max_new_tokens, int)
    return max_new_tokens


def _check_temperature_and_top_p(
    temperature: float,
    top_p: float,
    supports_zero_temperature: bool = True,
    supports_zero_top_p: bool = True,
    supports_one_top_p: bool = True,
) -> tuple[float, float]:
    if not supports_zero_temperature and temperature == 0.0:
        temperature = 1.0
        top_p = 0.0
    if not supports_zero_top_p and top_p == 0.0:
        temperature = 1.0
        top_p = 0.001
    if not supports_one_top_p and top_p == 1.0:
        top_p = 0.999
    return temperature, top_p


class LLM(_Cachable):
    def __init__(self, cache_folder_path: None | str = None):
        """Base class for all LLMs.

        Args:
            cache_folder_path: The path to the cache folder. If ``None``, the default
                cache folder for the DataDreamer session will be used.
        """
        super().__init__(cache_folder_path=cache_folder_path)
        self.chat_prompt_template: None | str = None
        self.system_prompt: None | str = None

    @abstractmethod
    def count_tokens(self, value: str) -> int:
        """Counts the number of tokens in a string.

        Args:
            value: The string to count tokens for.

        Returns:
            The number of tokens in the string.
        """
        pass

    def final_count_tokens(self, value: None | str) -> int:
        if value == "" or value is None:
            return 0
        else:
            return self.count_tokens(value)

    @cached_property
    def _chat_prompt_template_token_count(self) -> int:
        if self.chat_prompt_template:
            chat_prompt = self.chat_prompt_template.replace(
                "{{system_prompt}}", self.system_prompt or ""
            )
            return self.final_count_tokens(chat_prompt) - self.final_count_tokens(
                "{{prompt}}"
            )
        else:
            return 0

    @abstractmethod
    def get_max_context_length(self, max_new_tokens: int) -> int:
        """Gets the maximum context length for the model. When ``max_new_tokens`` is
        greater than 0, the maximum number of tokens that can be used for the prompt
        context is returned.

        Args:
            max_new_tokens: The maximum number of tokens that can be generated.

        Returns:
            The maximum context length.
        """
        pass

    def _get_max_output_length(self) -> None | int:
        """Gets the maximum output length for the model. If there is no maximum output
        limit and the only limit is the context length of the model, ``None`` is
        returned.

        Returns:
            The maximum output length.
        """
        return None

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
        """Formats a prompt for the LLM given instructions and in-context examples.

        .. dropdown:: Prompt Format

            The final prompt will be constructed as follows:

            .. code-block:: python

                beg_instruction
                sep
                in_context_example_1
                sep
                in_context_example_2
                sep
                ...
                sep
                in_context_example_n
                sep
                end_instruction

            If ``beg_instruction``, ``in_context_examples``, and ``end_instruction`` are
            ``None``, they will not be included in the prompt.

            If all of the ``in_context_examples`` will not fit in the prompt
            (accounting for the possible ``max_new_tokens`` that may be generated) the
            prompt will be constructed with as many in-context examples that will fit.

            If ``min_in_context_examples`` and ``max_in_context_examples`` are set,
            those constraints will be enforced.

        Args:
            max_new_tokens: The maximum number of tokens that can be generated.
            beg_instruction: The instruction at the beginning of the prompt.
            in_context_examples: The in-context examples to include in the prompt.
            end_instruction: The instruction at the end of the prompt.
            sep: The separator to use between the instructions and in-context examples.
            min_in_context_examples: The minimum number of in-context examples to include
                in the prompt.
            max_in_context_examples: The maximum number of in-context examples to include
                in the prompt.

        :rtype: :py:class:`str`

        Returns:
            The formatted prompt.
        """

        # Set max_new_tokens
        if max_new_tokens is None:
            max_new_tokens = 0
        assert max_new_tokens is not None

        # Initialize in_context_examples
        if in_context_examples is None:
            in_context_examples = []
        else:
            in_context_examples = in_context_examples.copy()
        assert in_context_examples is not None

        # Set min, max in-context examples
        if len(in_context_examples) > 0:
            if min_in_context_examples is None:
                min_in_context_examples = 1
            if max_in_context_examples is None:
                max_in_context_examples = len(in_context_examples)

        # Validate min, max in-context examples
        if (
            len(in_context_examples) > 0
            and min_in_context_examples is not None
            and max_in_context_examples is not None
        ):
            assert (
                min_in_context_examples >= 0
            ), "min_in_context_examples cannot be negative"
            assert (
                max_in_context_examples >= min_in_context_examples
            ), "max_in_context_examples cannot be less than min_in_context_examples"
            assert (
                len(in_context_examples) >= min_in_context_examples
            ), "len(in_context_examples) cannot be less than min_in_context_examples"

        # Get the max context length
        max_context_length = (
            self.get_max_context_length(max_new_tokens=max_new_tokens)
            - self._chat_prompt_template_token_count
        )

        # Define final prompt construction function
        def construct_final_prompt(in_context_examples):
            return sep.join(
                chain.from_iterable(
                    [
                        ([beg_instruction] if beg_instruction is not None else []),
                        in_context_examples,
                        ([end_instruction] if end_instruction is not None else []),
                    ]
                )
            )

        # Get the minimum required token count for instructions
        required_token_count = self.final_count_tokens(construct_final_prompt([]))

        # Get how many tokens are left for in-context examples
        remaining_token_count = max_context_length - required_token_count
        if (
            min_in_context_examples is not None
            and min_in_context_examples >= 0
            and remaining_token_count <= self.final_count_tokens(sep)
        ):
            raise ValueError(
                "The prompt's template exceeds the max context length of the model"
                " and there is no room for in-context examples."
            )
        elif remaining_token_count < 0:
            raise ValueError(
                "The prompt's template exceeds the max context length of the model."
            )

        # Remove any in-context examples that would not fit
        in_context_examples_filtered: list[str] = []
        for ice in in_context_examples:
            # Calculate how many tokens, *if* this in-context example is added
            # to the prompt
            next_sum = self.final_count_tokens(
                construct_final_prompt(in_context_examples_filtered + [ice])
            )

            # If there is room to add this in-context example to the prompt, add it
            if next_sum <= max_context_length:
                in_context_examples_filtered.append(ice)
                if (
                    max_in_context_examples is not None
                    and len(in_context_examples_filtered) >= max_in_context_examples
                ):
                    break
        if (
            min_in_context_examples is not None
            and len(in_context_examples_filtered) < min_in_context_examples
        ):
            raise ValueError(
                f"Cannot fit the minimum {min_in_context_examples}"
                " in-context examples in the prompt without exceeding"
                " the max context length of the model."
            )

        # Construct the final prompt
        final_prompt = construct_final_prompt(in_context_examples_filtered)
        return final_prompt

    def _run_over_batches(  # noqa: C901
        self,
        run_batch: Callable[..., list[Any]],
        get_max_input_length_function: None | Callable[[], dict[str, Any]],
        max_model_length: None | int | Callable,
        inputs: Iterable[Any],
        batch_size: int = 1,
        batch_scheduler_buffer_size: None | int = None,
        adaptive_batch_size: bool = True,
        progress_interval: None | int = 60,
        force: bool = False,
        cache_only: bool = False,
        verbose: None | bool = None,
        log_level: None | int = None,
        total_num_inputs: None | int = None,
        **kwargs,
    ) -> Generator[Any, None, None]:
        # Apply a chat prompt template over the inputs if there is one
        if self.chat_prompt_template:

            def apply_chat_prompt_template(prompt: str) -> str:
                applied = (
                    cast(str, self.chat_prompt_template)
                    .replace("{{system_prompt}}", self.system_prompt or "")
                    .replace("{{prompt}}", prompt)
                )
                if isinstance(prompt, _StrWithSeed):  # pragma: no cover
                    return _StrWithSeed(applied, seed=prompt)
                else:
                    return applied

            inputs = map(apply_chat_prompt_template, inputs)

        return super()._run_over_batches(
            run_batch=run_batch,
            get_max_input_length_function=get_max_input_length_function,
            max_model_length=max_model_length,
            inputs=inputs,
            batch_size=batch_size,
            batch_scheduler_buffer_size=batch_scheduler_buffer_size,
            adaptive_batch_size=adaptive_batch_size,
            progress_interval=progress_interval,
            force=force,
            cache_only=cache_only,
            verbose=verbose,
            log_level=log_level,
            total_num_inputs=total_num_inputs,
            **kwargs,
        )

    @abstractmethod
    def run(
        self,
        prompts: Iterable[str],
        max_new_tokens: None | int = None,
        temperature: float = 1.0,
        top_p: float = 0.0,
        n: int = 1,
        stop: None | str | list[str] = None,
        repetition_penalty: None | float = None,
        logit_bias: None | dict[int, float] = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        batch_scheduler_buffer_size: None | int = None,
        adaptive_batch_size: bool = False,
        seed: None | int = None,
        progress_interval: None | int = 60,
        force: bool = False,
        cache_only: bool = False,
        verbose: None | bool = None,
        log_level: None | int = None,
        total_num_prompts: None | int = None,
        return_generator: bool = False,
        **kwargs,
    ) -> Generator[str | list[str], None, None] | list[str | list[str]]:
        pass

    @cached_property
    def model_card(self) -> None | str:  # pragma: no cover
        return None

    @cached_property
    def license(self) -> None | str:  # pragma: no cover
        return None

    @cached_property
    def citation(self) -> None | list[str]:  # pragma: no cover
        return None

    @property
    def version(self) -> float:  # pragma: no cover
        return 1.0

    @cached_property
    def display_icon(self) -> str:
        return " 🧠 "

    @cached_property
    def display_name(self) -> str:
        return super().display_name

    @cached_property
    def _cache_name(self) -> None | str:  # pragma: no cover
        return None

    @property
    def _input_type(self) -> str:
        return "prompt"

    def __ring_key__(self) -> int:
        return uuid4().int

    def unload_model(self):  # pragma: no cover  # noqa: B027
        """Unloads resources required to run the LLM from memory."""
        pass


__all__ = ["LLM"]
