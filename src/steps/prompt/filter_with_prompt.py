import re

from ..data_card import DataCardType
from ..step import SuperStep, zipped
from ..step_operations import _INTERNAL_STEP_OPERATION_KEY
from ._prompt_base import _PromptBase
from .process_with_prompt import ProcessWithPrompt


class FilterWithPrompt(_PromptBase, SuperStep):
    """Filters a set of inputs using an instruction with a
    :py:class:`~datadreamer.llms.LLM` and ``filter_func`` which takes in a
    generation produced by the instruction and returns a ``bool`` on whether to
    keep the row or not."""

    def setup(self):
        self._register_prompt_inputs(prompt_input_type="input")
        self._register_prompt_args()
        self.register_arg(
            "instruction",
            required=True,
            help="An instruction that describes how to process the input.",
        )
        self.register_arg(
            "filter_func",
            required=False,
            help=("A function to filter the generations with."),
            default_help=("filtering by parsing 'Yes' / 'No' from the generation"),
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
        self.register_output(
            "*columns", help="All of the columns of the step producing the 'inputs'."
        )
        self.register_data_card(
            DataCardType.CITATION,
            """
@article{Zheng2023JudgingLW,
  title={Judging LLM-as-a-judge with MT-Bench and Chatbot Arena},
  author={Lianmin Zheng and Wei-Lin Chiang and Ying Sheng and Siyuan Zhuang and"""
            """ Zhanghao Wu and Yonghao Zhuang and Zi Lin and Zhuohan Li and Dacheng"""
            """ Li and Eric P. Xing and Haotong Zhang and Joseph Gonzalez and Ion"""
            """ Stoica},
  journal={ArXiv},
  year={2023},
  volume={abs/2306.05685},
  url={https://api.semanticscholar.org/CorpusID:259129398}
}
            """.strip(),
        )

    def run(self):
        # Get inputs and arguments
        args = self.args
        inputs = self.inputs["inputs"]
        lazy = args["lazy"]
        filter_func = args.pop("filter_func") or (
            lambda generation: re.search(r"\byes\W*\b", generation, re.IGNORECASE)
        )

        # Run the instruction on the inputs
        process_with_prompt = ProcessWithPrompt(
            name="Getting filter generations",
            inputs=self.inputs,
            args=args,
            outputs={"generations": "_filter_with_prompt_generations"},
        ).select_columns(
            ["_filter_with_prompt_generations"], name="Selecting filter generations"
        )

        # Combine the original step's rows with the filter generations
        orig_and_filter_generations = zipped(
            inputs.step,
            process_with_prompt,
            name="Combining rows with filter generations",
        )

        # Filter the original rows based on the filter generations
        return orig_and_filter_generations.map(
            name="Filtering rows",
            function=lambda rows: {
                k: col
                for k, col in rows.items()
                if k != "_filter_with_prompt_generations"
            }
            if filter_func(rows["_filter_with_prompt_generations"][0])
            else {k: [] for k in rows.keys() if k != "_filter_with_prompt_generations"},
            batched=True,
            batch_size=1,
            remove_columns=["_filter_with_prompt_generations"],
            lazy=lazy,
            total_num_rows=process_with_prompt.output[
                "_filter_with_prompt_generations"
            ].num_rows,
            auto_progress=False,
            progress_interval=self.progress_interval,
        ).output


setattr(FilterWithPrompt, _INTERNAL_STEP_OPERATION_KEY, True)

__all__ = ["FilterWithPrompt"]
