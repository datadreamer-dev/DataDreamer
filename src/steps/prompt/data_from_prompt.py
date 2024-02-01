from functools import partial

from ..._cachable._cachable import _StrWithSeed
from ._prompt_base import _PromptBase


class DataFromPrompt(_PromptBase):
    """Generates ``n`` rows of data using an instruction with a
    :py:class:`~datadreamer.llms.LLM`."""

    def setup(self):
        self._prompt_input_type = "none"
        self._register_prompt_args()
        self.register_arg(
            "instruction",
            required=True,
            help="The instruction to use to generate data.",
        )
        self.register_arg(
            "n", required=True, help="The number of rows to generate from the prompt."
        )
        self.register_arg(
            "temperature",
            required=False,
            default=1.0,
            help="The temperature to use when generating data.",
        )
        self.register_arg(
            "top_p",
            required=False,
            default=1.0,
            help="The top_p to use when generating data.",
        )
        self._register_prompt_optional_args()
        self._register_prompt_outputs()

    def run(self):
        # Get inputs and arguments
        args = self.args
        instruction = args.pop("instruction")
        n = args.pop("n")
        _seed = args.pop("_seed", None)

        def create_prompts(instruction, n, seed):
            for prompt_idx in range(n):
                yield _StrWithSeed(
                    instruction,
                    seed=((_seed, prompt_idx) if _seed is not None else prompt_idx),
                )

        return self._run_prompts(
            args=args,
            prompts=partial(create_prompts, instruction, n, _seed),
            total_num_prompts=n,
        )


__all__ = ["DataFromPrompt"]
