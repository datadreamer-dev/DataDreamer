from collections import Counter
from functools import partial
from itertools import cycle, product, repeat

from ..._cachable._cachable import _StrWithSeed
from ...utils.str_utils import get_templated_var_names, replace_templated_vars
from ..data_card import DataCardType
from ._prompt_base import _PromptBase


class DataFromAttributedPrompt(_PromptBase):
    """Generates ``n`` rows of data using an attributed instruction with a
    :py:class:`~datadreamer.llms.LLM`. See the
    `AttrPrompt paper <https://arxiv.org/abs/2306.15895>`_
    for more information.

    .. dropdown:: Format of the ``instruction`` and ``attributes`` arguments

        The ``instruction`` argument is a string with templated variables representing
        attributes, for example:

        .. code-block:: python

            instruction = "Generate a {adjective} sentence that is {length}."

        Then, the ``attributes`` argument is a dictionary of lists, where the keys are
        the attribute names and the values are the possible values for the
        attribute, for example:

        .. code-block:: python

            attributes = {
                "adjective": ["serious", "funny"],
                "length": ["short", "long"],
            }

        So all combinations of attributes will be used to generate data, by replacing
        the templated variables in the instruction with the attribute values to create 4
        distinct attributed prompts to use:

        1. "Generate a serious sentence that is short."
        2. "Generate a serious sentence that is long."
        3. "Generate a funny sentence that is short."
        4. "Generate a funny sentence that is long."

        If you want to directly specify the combinations of attributes, without
        automatically using all possible combinations, you can pass in a list of
        dictionaries to the ``attributes`` argument instead. Then you can directly
        specify which combinations should be used to create the attributed prompts, for
        example:

        .. code-block:: python

                attributes = [
                    {"adjective": "serious", "length": "short"},
                    {"adjective": "funny", "length": "short"},
                    {"adjective": "funny", "length": "long"},
                ]

        With this specification of ``attributes``, only 3 attributed prompts will be
        used instead of 4.
    """

    def setup(self):
        self._prompt_input_type = "none"
        self._register_prompt_args()
        self.register_arg(
            "instruction",
            required=True,
            help="The attributed instruction to use to generate data.",
        )
        self.register_arg(
            "attributes",
            required=True,
            help="The attributes to use in the instruction.",
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
        self.register_output(
            "attributes", help="The attributes used to generate the data."
        )
        self._register_prompt_outputs()
        self.register_data_card(
            DataCardType.CITATION,
            """
@article{yu2023large,
  title={Large language model as attributed training data generator: A tale of"""
            """ diversity and bias},
  author={Yu, Yue and Zhuang, Yuchen and Zhang, Jieyu and Meng, Yu and Ratner,"""
            """ Alexander and Krishna, Ranjay and Shen, Jiaming and Zhang, Chao},
  journal={arXiv preprint arXiv:2306.15895},
  year={2023}
}
            """.strip(),
        )

    def run(self):
        # Get inputs and arguments
        args = self.args
        instruction = args.pop("instruction")
        attributes = args.pop("attributes")
        n = args.pop("n")
        _seed = args.pop("_seed", None)

        # Get all templated attribute names in the instruction
        attribute_names = get_templated_var_names(instruction)

        # Get all combinations of attributes
        assert isinstance(
            attributes, (list, dict)
        ), f"Invalid type provided for `attributes`: {type(attributes)}"

        def get_attribute_combinations(attributes):
            if isinstance(attributes, dict):
                attribute_combinations = cycle(
                    enumerate(
                        map(
                            lambda comb: dict(comb),
                            product(
                                *[zip(repeat(k), v) for k, v in attributes.items()]
                            ),
                        )
                    )
                )
            else:
                attribute_combinations = cycle(enumerate(attributes))
            return attribute_combinations

        def create_prompts(
            instruction, attribute_names, attribute_combinations, n, _seed
        ):
            counter: Counter[int] = Counter()
            for _, (comb_idx, attribute_combination) in zip(
                range(n), attribute_combinations
            ):
                # Check the attribute combination to see if it is valid
                assert isinstance(attribute_combination, dict), (
                    f"Expected a dictionary of attributes. Got"
                    f" {type(attribute_combination)}"
                )
                assert set(attribute_combination.keys()) == set(attribute_names), (
                    f"Expected attribute names ({attribute_names}) from the"
                    f" instruction. Got: {(list(attribute_combination.keys()))}"
                )
                attributed_instruction = replace_templated_vars(
                    instruction, attribute_combination
                )
                yield _StrWithSeed(
                    attributed_instruction,
                    seed=(
                        (_seed, counter[comb_idx])
                        if _seed is not None
                        else counter[comb_idx]
                    ),
                )
                counter[comb_idx] += 1

        def extra_columns():
            for _, attribute_combination in get_attribute_combinations(attributes):

                def get_final_row(attribute_combination, row):
                    return {
                        "attributes": attribute_combination,
                        "prompts": row["prompts"],
                        "generations": row["generations"],
                    }

                yield partial(get_final_row, attribute_combination)

        return self._run_prompts(
            args=args,
            prompts=partial(
                create_prompts,
                instruction,
                attribute_names,
                get_attribute_combinations(attributes),
                n,
                _seed,
            ),
            total_num_prompts=n,
            extra_columns=extra_columns,
        )


__all__ = ["DataFromAttributedPrompt"]
