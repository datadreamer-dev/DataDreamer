from functools import cached_property
from itertools import tee
from typing import Any, Generator, Iterable, cast

import torch
import torch.nn.functional as F
from datasets.features.features import Value

from ..._cachable import _Cachable, _ParallelCachable
from ..data_card import DataCardType
from ..step import SuperStep
from ..step_output import LazyRows
from .embed import Embed

_DEFAULT_BATCH_SIZE = 100


class _CosineSimilarityComputer(_Cachable):
    def __init__(
        self,
        device: None | int | str | torch.device = None,
        cache_folder_path: None | str = None,
    ):
        super().__init__(cache_folder_path=cache_folder_path)
        self.device = device

    @torch.no_grad()
    def _run_batch(  # noqa: C901
        self, inputs: list[str], batch_size: int = _DEFAULT_BATCH_SIZE, **kwargs
    ) -> list[float]:
        a_and_b = list(inputs)
        a = torch.tensor(list(map(lambda row: row[0], a_and_b)), device=self.device)
        b = torch.tensor(list(map(lambda row: row[1], a_and_b)), device=self.device)
        sims = F.cosine_similarity(a, b, dim=1).detach().cpu().tolist()
        return sims

    def run(
        self,
        a_and_b: Iterable[tuple[Any, Any]],
        batch_size: int = _DEFAULT_BATCH_SIZE,
        batch_scheduler_buffer_size: None | int = None,
        adaptive_batch_size: bool = True,
        progress_interval: None | int = 60,
        force: bool = False,
        cache_only: bool = False,
        verbose: None | bool = None,
        log_level: None | int = None,
        total_num_embeddings: None | int = None,
        return_generator: bool = False,
        **kwargs,
    ) -> Generator[float, None, None] | list[float]:
        results_generator = self._run_over_batches(
            run_batch=self._run_batch,
            get_max_input_length_function=None,  # Not supported
            max_model_length=None,  # Not supported
            inputs=a_and_b,
            batch_size=batch_size,
            batch_scheduler_buffer_size=batch_size,  # Not supported
            adaptive_batch_size=False,  # Not supported
            progress_interval=progress_interval,
            force=force,
            cache_only=cache_only,
            verbose=verbose,
            log_level=log_level,
            total_num_inputs=total_num_embeddings,
            **kwargs,
        )
        if not return_generator:  # pragma: no cover
            return list(results_generator)
        else:
            return results_generator

    @property
    def version(self) -> float:  # pragma: no cover
        return 1.0

    @cached_property
    def display_name(self) -> str:
        return "Compute Cosine Similarities"

    @property
    def _input_type(self) -> str:
        return "embedding"


class _ParallelCosineSimilarityComputer(_ParallelCachable, _CosineSimilarityComputer):
    def __init__(self, *computers: _CosineSimilarityComputer):
        super().__init__(*computers, cls=_CosineSimilarityComputer)
        self.computers = cast(list[_CosineSimilarityComputer], self.cachables)

    def run(
        self, a_and_b: Iterable[tuple[Any, Any]], *args, **kwargs
    ) -> Generator[float, None, None] | list[float]:
        kwargs["batch_size"] = kwargs.pop("batch_size", _DEFAULT_BATCH_SIZE)
        results_generator = self._run_in_parallel(a_and_b, *args, **kwargs)
        if not kwargs.get("return_generator", False):  # pragma: no cover
            return list(results_generator)
        else:
            return results_generator


class CosineSimilarity(SuperStep):
    """Computes the cosine similarity between two sets of embeddings (``a`` and ``b``).
    If ``a`` and ``b`` are a set of texts, then they will be embedded using the provided
    :py:class:`~datadreamer.embedders.Embedder`."""

    def setup(self):
        self.register_input(
            "a", help="The embeddings or texts to compute the cosine similarity of."
        )
        self.register_input(
            "b", help="The embeddings or texts to compute the cosine similarity of."
        )
        self.register_arg(
            "similarity_batch_size",
            required=False,
            default=_DEFAULT_BATCH_SIZE,
            help="How many cosine similarities to compute at once.",
        )
        self.register_arg(
            "device",
            required=False,
            help="The device or list of devices to compute the cosine similarities"
            " with.",
        )
        self.register_arg(
            "embedder",
            required=False,
            help="The Embedder to use to embed 'a' and 'b' if they are a set of texts.",
        )
        self.register_arg(
            "lazy", required=False, default=False, help="Whether to run lazily or not."
        )
        self.register_arg(
            "**kwargs",
            required=False,
            help="Any other arguments you want to pass to the .run() method of the"
            " Embedder if it is used.",
        )
        self.register_output(
            "a",
            help="The embeddings or texts that the cosine similarities were computed"
            " for.",
        )
        self.register_output(
            "b",
            help="The embeddings or texts that the cosine similarities were computed"
            " for.",
        )
        self.register_output(
            "similarities", help="The similarities computed by the step."
        )

    def run(self):
        args = self.args

        # Get arguments
        similarity_batch_size = args.pop("similarity_batch_size")
        device = args.pop("device", None)
        embedder = args.pop("embedder", None)
        lazy = args.pop("lazy")

        # Get the a and b inputs
        a = self.inputs["a"]
        b = self.inputs["b"]
        total_num_inputs = a.num_rows

        # Embed a and b if they are provided as texts
        embeddings = {}
        for col_name, col in {"a": a, "b": b}.items():
            col_feature = col._features.get(col.column_names[0], None)
            if isinstance(col_feature, Value) and col_feature.dtype == "string":
                assert (
                    embedder is not None
                ), "You must provide an embedder if 'a' or 'b' are texts."
                embeddings[col_name] = Embed(
                    f"Embed '{col_name}' texts",
                    inputs={"texts": col},
                    args={"embedder": embedder, "lazy": lazy, **args},
                ).output["embeddings"]
            else:
                embeddings[col_name] = col

        assert (
            embedder is None or (embeddings["a"] != a or embeddings["b"] != b)
        ), "You should not provide `embedder` if both 'a' and 'b' are already embeddings."

        # Register trace info from the Embedder model
        if embedder is not None:
            if hasattr(embedder, "model_name"):
                self.register_data_card(DataCardType.MODEL_NAME, embedder.model_name)
            self.register_data_card(DataCardType.MODEL_CARD, embedder.model_card)
            self.register_data_card(DataCardType.LICENSE, embedder.license)
            for citation in embedder.citation or []:
                self.register_data_card(DataCardType.CITATION, citation)

        # Define a function that yields similarities
        def get_similarities():
            # Get an iterator over embeddings
            if embeddings["a"] == a:
                a_iter, a_embeddings_iter = tee(iter(a), 2)
            else:
                a_iter, a_embeddings_iter = iter(a), iter(embeddings["a"])
            if embeddings["b"] == b:
                b_iter, b_embeddings_iter = tee(iter(b), 2)
            else:
                b_iter, b_embeddings_iter = iter(b), iter(embeddings["b"])

            # Compute similarities
            compute_similarities_args = args.copy()
            compute_similarities_args.pop("batch_size", None)
            compute_similarities_args.pop("truncate", None)
            compute_similarities_args.pop("instruction", None)
            cosine_similarity_computer: _CosineSimilarityComputer
            if isinstance(device, list):
                cosine_similarity_computer = _ParallelCosineSimilarityComputer(
                    *[_CosineSimilarityComputer(device=d) for d in device]
                )
            else:
                cosine_similarity_computer = _CosineSimilarityComputer(device=device)
            similarities_iter = iter(
                cosine_similarity_computer.run(
                    a_and_b=zip(a_embeddings_iter, b_embeddings_iter),
                    batch_size=similarity_batch_size,
                    progress_interval=self.progress_interval,
                    total_num_texts=total_num_inputs,
                    return_generator=True,
                    _step=self,
                    **compute_similarities_args,
                )
            )

            yield from zip(a_iter, b_iter, similarities_iter)

        # Return embeddings
        return LazyRows(
            get_similarities,
            total_num_rows=total_num_inputs,
            auto_progress=False,
            save=(not lazy),
        )


__all__ = ["CosineSimilarity"]
