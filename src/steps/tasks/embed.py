from itertools import tee

from ..data_card import DataCardType
from ..step import Step
from ..step_output import LazyRows


class Embed(Step):
    "Embeds a set of texts with an :py:class:`~datadreamer.embedders.Embedder`."

    def setup(self):
        self.register_input("texts", help="The texts to embed.")
        self.register_arg("embedder", help="The Embedder to use.")
        self.register_arg(
            "truncate",
            required=False,
            default=False,
            help="Whether or not to truncate inputs.",
        )
        self.register_arg(
            "instruction",
            required=False,
            help="The instruction to prefix inputs to the embedding model with.",
        )
        self.register_arg(
            "lazy", required=False, default=False, help="Whether to run lazily or not."
        )
        self.register_arg(
            "**kwargs",
            required=False,
            help="Any other arguments you want to pass to the .run() method of the Embedder.",
        )
        self.register_output("texts", help="The texts that were embedded.")
        self.register_output("embeddings", help="The embeddings by the Embedder.")

    def run(self):
        args = self.args

        # Get inputs and arguments
        embedder = args.pop("embedder")
        lazy = args.pop("lazy")

        # Register trace info from the Embedder model
        if hasattr(embedder, "model_name"):
            self.register_data_card(DataCardType.MODEL_NAME, embedder.model_name)
        self.register_data_card(DataCardType.MODEL_CARD, embedder.model_card)
        self.register_data_card(DataCardType.LICENSE, embedder.license)
        for citation in embedder.citation or []:
            self.register_data_card(DataCardType.CITATION, citation)

        # Get the total number of texts
        texts = self.inputs["texts"]
        total_num_texts = texts.num_rows

        # Define a function that yields embeddings
        def get_embeddings():
            # Get an iterator over texts
            texts_iter_1, texts_iter_2 = tee(iter(texts), 2)

            # Generate
            embeddings_iter = iter(
                embedder.run(
                    texts=texts_iter_1,
                    progress_interval=self.progress_interval,
                    total_num_texts=total_num_texts,
                    return_generator=True,
                    _step=self,
                    **args,
                )
            )

            yield from zip(texts_iter_2, embeddings_iter)

        # Return embeddings
        return LazyRows(
            get_embeddings,
            total_num_rows=total_num_texts,
            auto_progress=False,
            save=(not lazy),
        )


__all__ = ["Embed"]
