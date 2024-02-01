from itertools import tee

from ...retrievers import EmbeddingRetriever
from ..data_card import DataCardType
from ..step import Step
from ..step_output import LazyRows


class Retrieve(Step):
    """Retrieves results for a set of queries with a
    :py:class:`~datadreamer.retrievers.Retriever`."""

    def setup(self):
        self.register_input("queries", help="The queries to retrieve results for.")
        self.register_arg("retriever", help="The Retriever to use.")
        self.register_arg(
            "k", required=False, default=5, help="How many results to retrieve."
        )
        self.register_arg(
            "lazy", required=False, default=False, help="Whether to run lazily or not."
        )
        self.register_arg(
            "**kwargs",
            required=False,
            help="Any other arguments you want to pass to the .run() method of the Retriever.",
        )
        self.register_output("queries", help="The queries used to retrieve results.")
        self.register_output("results", help="The results from the Retriever.")

    def run(self):
        args = self.args

        # Get inputs and arguments
        retriever = args.pop("retriever")
        lazy = args.pop("lazy")

        # Register trace info from the Retriever model
        if hasattr(retriever, "model_name"):  # pragma: no cover
            self.register_data_card(DataCardType.MODEL_NAME, retriever.model_name)
        self.register_data_card(DataCardType.MODEL_CARD, retriever.model_card)
        self.register_data_card(DataCardType.LICENSE, retriever.license)
        for citation in retriever.citation or []:
            self.register_data_card(DataCardType.CITATION, citation)
        if isinstance(retriever, EmbeddingRetriever):
            if hasattr(retriever.embedder, "model_name"):
                self.register_data_card(
                    DataCardType.MODEL_NAME, retriever.embedder.model_name
                )
            self.register_data_card(
                DataCardType.MODEL_CARD, retriever.embedder.model_card
            )
            self.register_data_card(DataCardType.LICENSE, retriever.embedder.license)
            for citation in retriever.embedder.citation or []:
                self.register_data_card(DataCardType.CITATION, citation)

        # Get the total number of queries
        queries = self.inputs["queries"]
        total_num_queries = queries.num_rows

        # Define a function that yields results
        def get_results():
            # Get an iterator over queries
            queries_iter_1, queries_iter_2 = tee(iter(queries), 2)

            # Generate
            results_iter = iter(
                retriever.run(
                    queries=queries_iter_1,
                    progress_interval=self.progress_interval,
                    total_num_queries=total_num_queries,
                    return_generator=True,
                    _step=self,
                    **args,
                )
            )

            yield from zip(queries_iter_2, results_iter)

        # Return results
        return LazyRows(
            get_results,
            total_num_rows=total_num_queries,
            auto_progress=False,
            save=(not lazy),
        )


__all__ = ["Retrieve"]
