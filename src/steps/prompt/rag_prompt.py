from functools import partial

from ..data_card import DataCardType
from ..step import SuperStep
from ..tasks.retrieve import Retrieve
from ._prompt_base import _PromptBase


class RAGPrompt(_PromptBase, SuperStep):
    """Processes a set of prompts using in-context texts with a
    :py:class:`~datadreamer.llms.LLM`. The ``k`` most relevant texts are
    retrieved for each prompt using a :py:class:`~datadreamer.retrievers.Retriever`."""

    def setup(self):
        self._register_prompt_inputs()
        self._register_prompt_args()
        self.register_arg(
            "retriever",
            required=True,
            help="The retriever to use to retrieve in-context texts.",
        )
        self.register_arg("k", required=True, help="The number of texts to retrieve.")
        self.register_arg(
            "retrieved_text_label",
            required=False,
            default="Document:",
            help="The label to use for retrieved texts.",
        )
        self.register_arg(
            "prompt_label",
            required=False,
            default="Question:",
            help="The label to use for the prompt.",
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
            help=(
                "The separator to use between in-context retrieved texts and each"
                " prompt."
            ),
        )
        self.register_arg(
            "min_in_context_retrieved_texts",
            required=False,
            help="The minimum number of in-context retrieved texts to include.",
        )
        self.register_arg(
            "max_in_context_retrieved_texts",
            required=False,
            help="The maximum number of in-context retrieved texts to include.",
        )
        self._register_prompt_optional_args()
        self._register_prompt_outputs()
        self.register_data_card(
            DataCardType.CITATION,
            """
@article{Guu2020REALMRL,
  title={REALM: Retrieval-Augmented Language Model Pre-Training},
  author={Kelvin Guu and Kenton Lee and Zora Tung and Panupong Pasupat and Ming-Wei"""
            """ Chang},
  journal={ArXiv},
  year={2020},
  volume={abs/2002.08909},
  url={https://api.semanticscholar.org/CorpusID:211204736}
}
            """.strip(),
        )

    def _in_context_retrieved_texts_generator(self):
        # Get inputs and arguments
        prompts = self.inputs["prompts"]
        retriever = self.args["retriever"]
        k = self.args["k"]
        lazy = self.args["lazy"]

        # Retrieve the best texts for each prompt
        retrieval_results = Retrieve(
            "Retrieve most relevant in-context texts",
            args={"retriever": retriever, "k": k, "lazy": lazy},
            inputs={"queries": prompts},
        )

        # Create generators for the retrieved texts for each prompt
        def retrieved_texts_generator():
            for _, retrieval_result in zip(
                prompts, retrieval_results.output["results"]
            ):
                yield retrieval_result["texts"]

        return retrieved_texts_generator

    def run(self):
        # Get inputs and arguments
        args = self.args
        llm = args["llm"]
        prompts = self.inputs["prompts"]
        retrieved_text_label = args.pop("retrieved_text_label")
        prompt_label = args.pop("prompt_label")
        max_new_tokens = args["max_new_tokens"]
        format_prompt_args = dict(
            max_new_tokens=max_new_tokens,
            beg_instruction=args.pop("beg_instruction", None),
            sep=args.pop("sep"),
            min_in_context_examples=args.pop("min_in_context_retrieved_texts"),
            max_in_context_examples=args.pop("max_in_context_retrieved_texts"),
        )
        if format_prompt_args["beg_instruction"] is not None:
            format_prompt_args["beg_instruction"] += format_prompt_args["sep"]

        # Create prompts with retrieved texts in-context and get an iterator over them
        retrieved_texts_generator = self._in_context_retrieved_texts_generator()

        def create_prompts(
            llm,
            retrieved_texts_generator,
            prompts,
            retrieved_text_label,
            prompt_label,
            format_prompt_args,
        ):
            for prompt, retrieved_texts in zip(prompts, retrieved_texts_generator()):
                in_context_retrieved_texts = [
                    (f"{retrieved_text_label} {text}" if retrieved_text_label else text)
                    for text in retrieved_texts
                ]
                end_instruction = format_prompt_args["sep"] + (
                    f"{prompt_label} {prompt}" if prompt_label else prompt
                )
                yield llm.format_prompt(
                    in_context_examples=in_context_retrieved_texts,
                    end_instruction=end_instruction,
                    **format_prompt_args,
                )

        # Generate
        return self._run_prompts(
            args=args,
            prompts=partial(
                create_prompts,
                llm,
                retrieved_texts_generator,
                prompts,
                retrieved_text_label,
                prompt_label,
                format_prompt_args,
            ),
        )


__all__ = ["RAGPrompt"]
