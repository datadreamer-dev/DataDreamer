from functools import partial

from ..data_card import DataCardType
from ._prompt_base import _PromptBase


class FewShotPrompt(_PromptBase):
    """Processes a set of inputs using in-context examples and an instruction with a
    :py:class:`~datadreamer.llms.LLM`."""

    def setup(self):
        self.register_input(
            "input_examples",
            help="The in-context example inputs to include in the prompt.",
        )
        self.register_input(
            "output_examples",
            help="The in-context example outputs to include in the prompt.",
        )
        self._register_prompt_inputs(prompt_input_type="input")
        self._register_prompt_args()
        self.register_arg(
            "input_label",
            required=False,
            default="Input:",
            help="The label to use for inputs.",
        )
        self.register_arg(
            "output_label",
            required=False,
            default="Output:",
            help="The label to use for outputs.",
        )
        self.register_arg(
            "max_new_tokens",
            required=False,
            help="The maximum number of tokens to generate.",
        )
        self.register_arg(
            "instruction",
            required=False,
            help="An instruction to include in the prompt.",
        )
        self.register_arg(
            "sep",
            required=False,
            default="\n",
            help="The separator to use between instructions and in-context examples.",
        )
        self.register_arg(
            "min_in_context_examples",
            required=False,
            help="The minimum number of in-context examples to include.",
        )
        self.register_arg(
            "max_in_context_examples",
            required=False,
            help="The maximum number of in-context examples to include.",
        )
        self._register_prompt_optional_args()
        self._register_prompt_outputs()
        self.register_data_card(
            DataCardType.CITATION,
            """
@inproceedings{Radford2019LanguageMA,
  title={Language Models are Unsupervised Multitask Learners},
  author={Alec Radford and Jeff Wu and Rewon Child and David Luan and Dario Amodei"""
            """ and Ilya Sutskever},
  year={2019},
  url={https://api.semanticscholar.org/CorpusID:160025533}
}
            """.strip(),
        )

    def _in_context_examples_generator(self):
        inputs = self.inputs["inputs"]
        input_examples = list(self.inputs["input_examples"])
        output_examples = list(self.inputs["output_examples"])
        assert len(input_examples) == len(
            output_examples
        ), "len(input_examples) must equal len(output_examples)"

        def input_examples_generator():
            for _ in inputs:
                yield input_examples

        def output_examples_generator():
            for _ in inputs:
                yield output_examples

        return input_examples_generator, output_examples_generator

    def run(self):
        # Get inputs and arguments
        args = self.args
        llm = args["llm"]
        input_examples = list(self.inputs["input_examples"])
        output_examples = list(self.inputs["output_examples"])
        inputs = self.inputs["inputs"]
        assert len(input_examples) == len(
            output_examples
        ), "len(input_examples) must equal len(output_examples)"
        input_label = args.pop("input_label")
        output_label = args.pop("output_label")
        max_new_tokens = args["max_new_tokens"]
        format_prompt_args = dict(
            max_new_tokens=max_new_tokens,
            beg_instruction=args.pop("instruction"),
            sep=args.pop("sep"),
            min_in_context_examples=args.pop("min_in_context_examples"),
            max_in_context_examples=args.pop("max_in_context_examples"),
        )
        if format_prompt_args["beg_instruction"] is not None:
            format_prompt_args["beg_instruction"] += format_prompt_args["sep"]

        # Create few-shot prompts and get an iterator over them
        (
            input_examples_generator,
            output_examples_generator,
        ) = self._in_context_examples_generator()

        def create_few_shot_prompts(
            llm,
            input_examples_generator,
            output_examples_generator,
            inputs,
            input_label,
            output_label,
            format_prompt_args,
        ):
            for input, input_examples, output_examples in zip(
                inputs, input_examples_generator(), output_examples_generator()
            ):
                in_context_examples = [
                    (f"{input_label} {input}" if input_label else input)
                    + format_prompt_args["sep"]
                    + (f"{output_label} {output}" if output_label else output)
                    for input, output in zip(input_examples, output_examples)
                ]
                end_instruction = (
                    (f"{input_label} {input}" if input_label else input)
                    + format_prompt_args["sep"]
                    + (f"{output_label}" if output_label else "")
                )
                yield llm.format_prompt(
                    in_context_examples=in_context_examples,
                    end_instruction=end_instruction,
                    **format_prompt_args,
                )

        # Generate
        return self._run_prompts(
            args=args,
            prompts=partial(
                create_few_shot_prompts,
                llm,
                input_examples_generator,
                output_examples_generator,
                inputs,
                input_label,
                output_label,
                format_prompt_args,
            ),
        )


__all__ = ["FewShotPrompt"]
