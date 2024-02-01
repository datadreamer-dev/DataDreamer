from functools import partial
from itertools import tee
from typing import Any, Callable, Generator, Iterable

from ..data_card import DataCardType
from ..step import Step
from ..step_output import LazyRows


class _PromptBase(Step):
    def _register_prompt_inputs(self, prompt_input_type: str = "prompt"):
        self._prompt_input_type = prompt_input_type
        self.register_input(
            f"{self._prompt_input_type}s",
            help=f"The {self._prompt_input_type}s to process with the LLM.",
        )

    def _register_prompt_args(self):
        self.register_arg("llm", help="The LLM to use.")

    def _register_prompt_optional_args(self):
        self.register_arg(
            "post_process",
            required=False,
            help="A function to post-process the generations.",
        )
        self.register_arg(
            "lazy", required=False, default=False, help="Whether to run lazily or not."
        )
        self.register_arg(
            "**kwargs",
            required=False,
            help="Any other arguments you want to pass to the .run() method of the LLM.",
        )

    def _register_prompt_outputs(self):
        if self._prompt_input_type == "input":
            self.register_output("inputs", help="The inputs processed with the LLM.")
        self.register_output("prompts", help="The prompts processed with the LLM.")
        self.register_output("generations", help="The generations by the LLM.")

    def _run_prompts(
        self,
        args: dict[str, Any],
        prompts: None | Iterable[str] | Callable[[], Generator[str, None, None]] = None,
        total_num_prompts: None | int = None,
        extra_columns: None
        | Callable[
            [], Generator[Callable[[dict[str, Any]], dict[str, Any]], None, None]
        ] = None,
    ):
        def default_extra_columns():
            while True:
                yield (lambda row: row)

        extra_columns = extra_columns or default_extra_columns

        # Get inputs and arguments
        llm = args.pop("llm")
        post_process = args.pop("post_process")
        lazy = args.pop("lazy")

        # Register trace info from the LLM model
        if hasattr(llm, "model_name"):
            self.register_data_card(DataCardType.MODEL_NAME, llm.model_name)
        self.register_data_card(DataCardType.MODEL_CARD, llm.model_card)
        self.register_data_card(DataCardType.LICENSE, llm.license)
        for citation in llm.citation or []:
            self.register_data_card(DataCardType.CITATION, citation)

        # Get the total number of prompts
        total_num_prompts = (
            total_num_prompts
            if total_num_prompts is not None
            else self.inputs[f"{self._prompt_input_type}s"].num_rows
        )

        # Define a function that yields generations
        def get_generations(prompts):
            # Get an iterator over prompts
            prompts = (
                prompts
                if prompts is not None
                else self.inputs[f"{self._prompt_input_type}s"]
            )
            if callable(prompts):
                prompts = prompts()
            prompts_iter_1, prompts_iter_2 = tee(iter(prompts), 2)

            # Generate
            generations_iter = iter(
                llm.run(
                    prompts=prompts_iter_1,
                    progress_interval=self.progress_interval,
                    total_num_prompts=total_num_prompts,
                    return_generator=True,
                    _step=self,
                    **args,
                )
            )
            if post_process is not None:
                generations_iter = map(post_process, generations_iter)

            if self._prompt_input_type == "input":
                for input, prompt, generation, get_extra_columns in zip(
                    self.inputs[f"{self._prompt_input_type}s"],
                    prompts_iter_2,
                    generations_iter,
                    extra_columns(),
                ):
                    yield get_extra_columns(
                        {"inputs": input, "prompts": prompt, "generations": generation}
                    )
            else:
                for prompt, generation, get_extra_columns in zip(
                    prompts_iter_2, generations_iter, extra_columns()
                ):
                    yield get_extra_columns(
                        {"prompts": prompt, "generations": generation}
                    )

        # Return generations
        return LazyRows(
            partial(get_generations, prompts),
            total_num_rows=total_num_prompts,
            auto_progress=False,
            save=(not lazy),
        )


__all__ = ["_PromptBase"]
