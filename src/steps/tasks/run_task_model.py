from itertools import tee

from ..data_card import DataCardType
from ..step import Step
from ..step_output import LazyRows


class RunTaskModel(Step):
    "Runs a set of texts against a :py:class:`~datadreamer.task_models.TaskModel`."

    def setup(self):
        self.register_input("texts", help="The texts to process with the TaskModel.")
        self.register_arg("model", help="The TaskModel to use.")
        self.register_arg(
            "truncate",
            required=False,
            default=False,
            help="Whether or not to truncate inputs.",
        )
        self.register_arg(
            "lazy", required=False, default=False, help="Whether to run lazily or not."
        )
        self.register_arg(
            "**kwargs",
            required=False,
            help="Any other arguments you want to pass to the .run() method of the TaskModel.",
        )
        self.register_output("texts", help="The texts processed with the TaskModel.")
        self.register_output("results", help="The results from the TaskModel.")

    def run(self):
        args = self.args

        # Get inputs and arguments
        model = args.pop("model")
        lazy = args.pop("lazy")

        # Register trace info from the TaskModel model
        if hasattr(model, "model_name"):
            self.register_data_card(DataCardType.MODEL_NAME, model.model_name)
        self.register_data_card(DataCardType.MODEL_CARD, model.model_card)
        self.register_data_card(DataCardType.LICENSE, model.license)
        for citation in model.citation or []:
            self.register_data_card(DataCardType.CITATION, citation)

        # Get the total number of texts
        texts = self.inputs["texts"]
        total_num_texts = texts.num_rows

        # Define a function that yields results
        def get_results():
            # Get an iterator over texts
            texts_iter_1, texts_iter_2 = tee(iter(texts), 2)

            # Generate
            results_iter = iter(
                model.run(
                    texts=texts_iter_1,
                    progress_interval=self.progress_interval,
                    total_num_texts=total_num_texts,
                    return_generator=True,
                    _step=self,
                    **args,
                )
            )

            yield from zip(texts_iter_2, results_iter)

        # Return results
        return LazyRows(
            get_results,
            total_num_rows=total_num_texts,
            auto_progress=False,
            save=(not lazy),
        )


__all__ = ["RunTaskModel"]
