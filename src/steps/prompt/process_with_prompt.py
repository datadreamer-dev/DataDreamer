from functools import partial

from ._prompt_base import _PromptBase


class ProcessWithPrompt(_PromptBase):
    """Processes a set of inputs using an instruction with a
    :py:class:`~datadreamer.llms.LLM`."""

    def setup(self):
        self._register_prompt_inputs(prompt_input_type="input")
        self._register_prompt_args()
        self.register_arg(
            "instruction",
            required=True,
            help="An instruction that describes how to process the input.",
        )
        self.register_arg(
            "input_label",
            required=False,
            default="Input:",
            help="The label to use for inputs.",
        )
        self.register_arg(
            "instruction_label",
            required=False,
            default="Instruction:",
            help="The label to use for the instruction.",
        )
        self.register_arg(
            "max_new_tokens",
            required=False,
            help="The maximum number of tokens to generate.",
        )
        self.register_arg(
            "sep",
            required=False,
            default="\n\n",
            help="The separator to use between instructions and the input.",
        )
        self._register_prompt_optional_args()
        self._register_prompt_outputs()

    def run(self):
        # Get inputs and arguments
        args = self.args
        llm = args["llm"]
        inputs = self.inputs["inputs"]
        input_label = args.pop("input_label")
        instruction_label = args.pop("instruction_label")
        max_new_tokens = args["max_new_tokens"]
        format_prompt_args = dict(
            max_new_tokens=max_new_tokens,
            end_instruction=(
                f"{instruction_label} {args.pop('instruction')}"
                if instruction_label
                else args.pop("instruction")
            ),
            sep=args.pop("sep"),
        )

        def create_process_input_with_instruction_prompts(
            llm, inputs, input_label, format_prompt_args
        ):
            for input in inputs:
                beg_instruction = f"{input_label} {input}" if input_label else input
                yield llm.format_prompt(
                    beg_instruction=beg_instruction, **format_prompt_args
                )

        # Generate
        return self._run_prompts(
            args=args,
            prompts=partial(
                create_process_input_with_instruction_prompts,
                llm,
                inputs,
                input_label,
                format_prompt_args,
            ),
        )


__all__ = ["ProcessWithPrompt"]
