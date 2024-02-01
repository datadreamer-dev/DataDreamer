from functools import partial
from random import Random

from ..data_card import DataCardType
from ._prompt_base import _PromptBase


class JudgeGenerationPairsWithPrompt(_PromptBase):
    """Judges between a of pair of generations (``a``, ``b``) that were produced in
    response to ``prompts`` using an instruction with a
    :py:class:`~datadreamer.llms.LLM`."""

    def setup(self):
        self._prompt_input_type = "prompt"
        self.register_input(
            "prompts", help=("A set of prompts used to generate 'a' and 'b'.")
        )
        self.register_input("a", help="A set of generations to judge between.")
        self.register_input("b", help="A set of generations to judge between.")
        self._register_prompt_args()
        self.register_arg(
            "instruction",
            required=False,
            help="An instruction that describes how to judge the input.",
            default_help=repr(
                "Which response is better? Respond with 'Response A' or 'Response B'."
            ),
        )
        self.register_arg(
            "judgement_func",
            required=False,
            help=("A function to get the judgement from the generation."),
            default_help=(
                "judging by parsing 'a_label' / 'b_label' ('Response A' / 'Response B')"
                " from the generation"
            ),
        )
        self.register_arg(
            "randomize_order",
            required=False,
            default=True,
            help=(
                "Whether to randomly swap the order of 'a' and 'b' in the prompt to"
                " mitigate position bias."
            ),
        )
        self.register_arg(
            "prompt_label",
            required=False,
            default="Prompt:",
            help="The label to use for prompts.",
        )
        self.register_arg(
            "a_label",
            required=False,
            default="Response A:",
            help="The label to use for inputs 'a'.",
        )
        self.register_arg(
            "b_label",
            required=False,
            default="Response B:",
            help="The label to use for inputs 'b'.",
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
            default="\n",
            help="The separator to use between instructions and the input.",
        )
        self._register_prompt_optional_args()
        self.register_output(
            "prompts", help=("The set of prompts used to generate 'a' and 'b'.")
        )
        self.register_output("a", help="A set of inputs judged with the LLM.")
        self.register_output("b", help="A set of inputs judged with the LLM.")
        self.register_output(
            "judge_prompts",
            help=("The prompts processed with the LLM to judge inputs."),
        )
        self.register_output(
            "judge_generations", help=("The judgement generations by the LLM.")
        )
        self.register_output("judgements", help="The judgements by the LLM.")
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
        args = self.args

        # Initialize random number generator
        randomize_order = args.pop("randomize_order")
        judgement_rng = Random(0)

        def randomize_order_rng():
            r = Random(1)
            while True:
                yield (r.randint(0, 1) == 1) if randomize_order else False

        # Get inputs and arguments
        llm = args["llm"]
        prompts = self.inputs["prompts"]
        a = self.inputs["a"]
        b = self.inputs["b"]
        prompt_label = args.pop("prompt_label")
        a_label = args.pop("a_label")
        b_label = args.pop("b_label")
        a_label_clean, b_label_clean = a_label.strip(":"), b_label.strip(":")
        instruction = args.pop("instruction")
        instruction = (
            instruction
            if instruction is not None
            else (
                f"Which response is better? Respond with '{a_label_clean}' or"
                f" '{b_label_clean}'."
            )
        )

        def default_judgement_func(gen, swapped):
            if a_label_clean in gen and b_label_clean not in gen:
                judgement = a_label_clean
            elif b_label_clean in gen and a_label_clean not in gen:
                judgement = b_label_clean
            else:
                judgement = (
                    b_label_clean if judgement_rng.randint(0, 1) else a_label_clean
                )
            if swapped:
                judgement = (
                    a_label_clean if judgement == b_label_clean else b_label_clean
                )
            return judgement

        judgement_func = args.pop("judgement_func") or default_judgement_func
        instruction_label = args.pop("instruction_label")
        max_new_tokens = args["max_new_tokens"]
        sep = args.pop("sep")
        format_prompt_args = dict(
            max_new_tokens=max_new_tokens,
            end_instruction=(
                sep
                + (
                    f"{instruction_label} {instruction}"
                    if instruction_label
                    else instruction
                )
            ),
            sep=sep,
        )

        def create_process_input_with_instruction_prompts(
            llm, prompts, a, b, prompt_label, a_label, b_label, format_prompt_args
        ):
            for prompt, each_a, each_b, should_swap in zip(
                prompts, a, b, randomize_order_rng()
            ):
                each_a, each_b = (each_b, each_a) if should_swap else (each_a, each_b)
                beg_instruction = (
                    (f"{prompt_label} {prompt}" if prompt_label else prompt)
                    + sep
                    + sep
                    + (
                        (f"{a_label} {each_a}" if a_label else each_a)
                        + sep
                        + sep
                        + (f"{b_label} {each_b}" if b_label else each_b)
                    )
                )
                yield llm.format_prompt(
                    beg_instruction=beg_instruction, **format_prompt_args
                )

        def extra_columns():
            for prompt, each_a, each_b, swapped in zip(
                prompts, a, b, randomize_order_rng()
            ):

                def get_final_row(prompt, each_a, each_b, swapped, row):
                    return {
                        "prompts": prompt,
                        "a": each_a,
                        "b": each_b,
                        "judge_prompts": row["prompts"],
                        "judge_generations": row["generations"],
                        "judgements": judgement_func(row["generations"], swapped),
                    }

                yield partial(get_final_row, prompt, each_a, each_b, swapped)

        # Generate
        return self._run_prompts(
            args=args,
            prompts=partial(
                create_process_input_with_instruction_prompts,
                llm,
                prompts,
                a,
                b,
                prompt_label,
                a_label,
                b_label,
                format_prompt_args,
            ),
            extra_columns=extra_columns,
        )


__all__ = ["JudgeGenerationPairsWithPrompt"]
