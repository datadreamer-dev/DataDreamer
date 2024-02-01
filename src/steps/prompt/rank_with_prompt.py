import re

from ..data_card import DataCardType
from ..step import SuperStep, zipped
from ..step_operations import _INTERNAL_STEP_OPERATION_KEY
from ._prompt_base import _PromptBase
from .process_with_prompt import ProcessWithPrompt


class RankWithPrompt(_PromptBase, SuperStep):
    """Produces scores for a set of inputs using an instruction with a
    :py:class:`~datadreamer.llms.LLM`. Scores are parsed from the generations produced
    by the instruction. Optionally, the input rows are sorted and filtered using the
    scores."""

    def setup(self):
        self._register_prompt_inputs(prompt_input_type="input")
        self._register_prompt_args()
        self.register_arg(
            "instruction",
            required=True,
            help="An instruction that describes how to process the input.",
        )
        self.register_arg(
            "sort",
            required=False,
            default=True,
            help=(
                "Whether or not to sort the inputs by the score produced by the"
                " instruction."
            ),
        )
        self.register_arg(
            "reverse",
            required=False,
            default=True,
            help=("Whether or not to reverse the sort direction."),
        )
        self.register_arg(
            "score_threshold",
            required=False,
            default=None,
            help=(
                "A score threshold. If specified, filter out input rows that scored"
                " below the threshold."
            ),
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
        self.register_output(
            "scores", help="The scores produced by the instruction on the inputs."
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
        sort = args.pop("sort")
        reverse = args.pop("reverse")
        post_process = args.pop("post_process") or (lambda x: x)
        score_threshold = args.pop("score_threshold")
        scores_col_name = self.output_name_mapping.pop("scores", "scores")

        # Run the instruction on the inputs
        def parse_num(generation):
            generation_post_processed = post_process(generation)
            match = re.search(r"[0-9\.]+", generation_post_processed)
            if match:
                return (
                    float(match.group()) if "." in match.group() else int(match.group())
                )
            else:  # pragma: no cover
                return None

        process_with_prompt = ProcessWithPrompt(
            name="Getting scores",
            inputs=self.inputs,
            args={"post_process": parse_num, **args},
            outputs={"generations": scores_col_name},
        ).select_columns([scores_col_name], name="Selecting scores")

        # Combine the original step's rows with the scores
        orig_and_filter_generations = zipped(
            inputs.step, process_with_prompt, name="Combining rows with scores"
        )

        # Sort the original rows based on the scores
        if sort:
            orig_and_filter_generations = orig_and_filter_generations.sort(
                scores_col_name,
                reverse=reverse,
                name="Sorting rows",
                progress_interval=self.progress_interval,
            )

        # Filter the original rows based on the filter generations
        if score_threshold is not None:
            orig_and_filter_generations = orig_and_filter_generations.filter(
                function=lambda row: row[scores_col_name] >= score_threshold,
                name=f"Filtering rows using score threshold: {score_threshold}",
                lazy=lazy,
                total_num_rows=process_with_prompt.output[scores_col_name].num_rows,
                auto_progress=False,
                progress_interval=self.progress_interval,
            )

        # Return the final output
        return orig_and_filter_generations.output


setattr(RankWithPrompt, _INTERNAL_STEP_OPERATION_KEY, True)

__all__ = ["RankWithPrompt"]
