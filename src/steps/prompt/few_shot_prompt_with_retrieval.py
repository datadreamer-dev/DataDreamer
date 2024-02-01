from ...datasets import OutputDatasetColumn
from ...retrievers import EmbeddingRetriever
from ..data_card import DataCardType
from ..step import SuperStep
from ..tasks.retrieve import Retrieve
from .few_shot_prompt import FewShotPrompt


class FewShotPromptWithRetrieval(FewShotPrompt, SuperStep):
    """Processes a set of inputs using in-context examples and an instruction with a
    :py:class:`~datadreamer.llms.LLM`. The ``k`` most relevant in-context examples are
    retrieved for each input using an :py:class:`~datadreamer.embedders.Embedder`."""

    def setup(self):
        self.register_input(
            "input_examples",
            help="The in-context example inputs to retrieve from to include in the prompt.",
        )
        self.register_input(
            "output_examples",
            help="The in-context example outputs to retrieve from to include in the prompt.",
        )
        self._register_prompt_inputs(prompt_input_type="input")
        self._register_prompt_args()
        self.register_arg(
            "embedder",
            required=True,
            help="The embedder to use to retrieve in-context examples.",
        )
        self.register_arg(
            "k", required=True, help="The number of in-context examples to retrieve."
        )
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
        self.register_data_card(
            DataCardType.CITATION,
            """
@inproceedings{shi-etal-2022-nearest,
    title = "Nearest Neighbor Zero-Shot Inference",
    author = "Shi, Weijia  and
      Michael, Julian  and
      Gururangan, Suchin  and
      Zettlemoyer, Luke",
    editor = "Goldberg, Yoav  and
      Kozareva, Zornitsa  and
      Zhang, Yue",
    booktitle = "Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing",
    month = dec,
    year = "2022",
    address = "Abu Dhabi, United Arab Emirates",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.emnlp-main.214",
    doi = "10.18653/v1/2022.emnlp-main.214",
    pages = "3254--3265"
}
            """.strip(),
        )

    def _in_context_examples_generator(self):
        # Get inputs and arguments
        inputs = self.inputs["inputs"]
        input_examples = self.inputs["input_examples"]
        output_examples = self.inputs["output_examples"]
        embedder = self.args["embedder"]
        k = self.args["k"]
        lazy = self.args["lazy"]
        assert isinstance(input_examples, OutputDatasetColumn) and isinstance(
            output_examples, OutputDatasetColumn
        ), "Input and output examples must not be iterable datasets."
        assert len(input_examples) == len(
            output_examples
        ), "len(input_examples) must equal len(output_examples)"

        # Retrieve the best examples for each input
        retriever = EmbeddingRetriever(texts=input_examples, embedder=embedder)
        retrieval_results = Retrieve(
            "Retrieve most relevant few-shot examples",
            args={"retriever": retriever, "k": k, "lazy": lazy},
            inputs={"queries": inputs},
        )

        # Create generators for the retrieved input and output examples for each input
        def input_examples_generator():
            for _, retrieval_result in zip(inputs, retrieval_results.output["results"]):
                yield retrieval_result["texts"]

        def output_examples_generator():
            for _, retrieval_result in zip(inputs, retrieval_results.output["results"]):
                yield [
                    output_examples[example_idx]
                    for example_idx in retrieval_result["indices"]
                ]

        return input_examples_generator, output_examples_generator


__all__ = ["FewShotPromptWithRetrieval"]
