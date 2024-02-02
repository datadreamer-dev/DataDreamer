import gc
from functools import cached_property, lru_cache, partial
from typing import Any, Callable, Generator, Iterable, cast

import numpy as np
import torch
import torch._dynamo
from datasets.fingerprint import Hasher

from ..logging import logger as datadreamer_logger
from ..task_models.task_model import DEFAULT_BATCH_SIZE, _check_texts_length
from ..utils.background_utils import RunIfTimeout
from ..utils.fs_utils import safe_fn
from ..utils.hf_hub_utils import (
    _has_file,
    get_citation_info,
    get_license_info,
    get_model_card_url,
)
from ..utils.hf_model_utils import (
    convert_dtype,
    get_model_max_context_length,
    get_tokenizer,
)
from ..utils.import_utils import ignore_transformers_warnings, import_module
from .embedder import Embedder

with ignore_transformers_warnings():
    from sentence_transformers import SentenceTransformer


@lru_cache(maxsize=None)
def _is_instructor_model(model_name: str):
    return "/instructor-" in model_name.lower()


@lru_cache(maxsize=None)
def _normalize_model_name(model_name: str) -> str:
    if "/" not in model_name:
        return f"sentence-transformers/{model_name}"
    else:
        return model_name


class SentenceTransformersEmbedder(Embedder):
    def __init__(
        self,
        model_name: str,
        trust_remote_code: bool = False,
        device: None | int | str | torch.device = None,
        dtype: None | str | torch.dtype = None,
        cache_folder_path: None | str = None,
        **kwargs,
    ):
        """Loads an `SentenceTransformers <https://www.sbert.net/>`_ embedder.

        Args:
            model_name: The name of the model to use.
            trust_remote_code: Whether to trust remote code.
            device: The device to use for the model.
            dtype: The type to use for the model weights.
            cache_folder_path: The path to the cache folder. If ``None``, the default
                cache folder for the DataDreamer session will be used.
            **kwargs: Additional keyword arguments to pass to the SentenceTransformers
                constructor.
        """
        super().__init__(model_name=model_name, cache_folder_path=cache_folder_path)
        self.hf_model_name = self.model_name
        if _has_file(
            repo_id=f"sentence-transformers/{self.model_name}",
            filename="config.json",
            repo_type="model",
        ):
            self.hf_model_name = f"sentence-transformers/{self.model_name}"
        self.trust_remote_code = trust_remote_code
        self.device = device
        self.dtype = convert_dtype(dtype)
        self.kwargs = kwargs

    @cached_property
    def model(self) -> SentenceTransformer:
        # Load model
        log_if_timeout = RunIfTimeout(
            partial(
                lambda self: self.get_logger(
                    key="model", log_level=datadreamer_logger.level
                ).info("Loading..."),
                self,
            ),
            timeout=10.0,
        )
        cls = SentenceTransformer
        if _is_instructor_model(self.model_name):  # pragma: no cover
            with ignore_transformers_warnings():
                cls = import_module("InstructorEmbedding").INSTRUCTOR
        model = cls(
            self.model_name,
            trust_remote_code=self.trust_remote_code,
            device=self.device,
            **self.kwargs,
        )
        model[0].tokenizer = get_tokenizer(
            _normalize_model_name(self.model_name),
            revision=None,
            trust_remote_code=False,
        )
        model.max_seq_length = (
            get_model_max_context_length(
                model_name=self.model_name, config=model[0].auto_model.config
            )
            if model.max_seq_length is None
            else model.max_seq_length
        )

        # Send model to accelerator device
        model = model.to(self.device)

        # Switch model to eval mode
        model.eval()

        # Torch compile
        # torch._dynamo.config.suppress_errors = True
        # model = torch.compile(model)

        # Finished loading
        log_if_timeout.stop(
            partial(
                lambda self: self.get_logger(
                    key="model", log_level=datadreamer_logger.level
                ).info("Finished loading."),
                self,
            )
        )

        return model

    @cached_property
    def tokenizer(self) -> Any:
        return self.model.tokenizer

    def count_tokens(self, value: str) -> int:
        """Counts the number of tokens in a string.

        Args:
            value: The string to count tokens for.

        Returns:
            The number of tokens in the string.
        """
        pass
        return len(self.tokenizer.encode(value))

    @cached_property
    def model_max_length(self) -> int:
        return self.model.max_seq_length

    @cached_property
    def dims(self) -> int:
        return self.model.get_sentence_embedding_dimension()

    @torch.no_grad()
    def _run_batch(
        self,
        max_length_func: Callable[[list[str]], int],
        inputs: list[str],
        truncate: bool = False,
        instruction: None | str = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        **kwargs,
    ) -> list[np.ndarray]:
        texts = inputs
        if not truncate:
            _check_texts_length(self=self, max_length_func=max_length_func, texts=texts)

        model_input: list[str] | list[list[str]] = texts
        if _is_instructor_model(self.model_name):  # pragma: no cover
            model_input = [[cast(str, instruction), t] for t in texts]

        return list(
            self.model.encode(
                sentences=model_input,
                batch_size=len(texts),
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=kwargs.pop("normalize_embeddings", True),
                **kwargs,
            )
        )

    def _run_over_batches(  # noqa: C901
        self,
        run_batch: Callable[..., list[Any]],
        get_max_input_length_function: None | Callable[[], dict[str, Any]],
        max_model_length: None | int | Callable,
        inputs: Iterable[Any],
        batch_size: int = 1,
        batch_scheduler_buffer_size: None | int = None,
        adaptive_batch_size: bool = True,
        progress_interval: None | int = 60,
        force: bool = False,
        cache_only: bool = False,
        verbose: None | bool = None,
        log_level: None | int = None,
        total_num_inputs: None | int = None,
        **kwargs,
    ) -> Generator[Any, None, None]:
        # Apply an instruction over the inputs if there is one
        if kwargs.get("instruction", None) is not None and not _is_instructor_model(
            self.model_name
        ):
            instruction = kwargs["instruction"]

            def apply_instruction(instruction: str, text: str) -> str:
                return instruction + text

            inputs = map(partial(apply_instruction, instruction), inputs)

        return super()._run_over_batches(
            run_batch=run_batch,
            get_max_input_length_function=get_max_input_length_function,
            max_model_length=self.model_max_length,
            inputs=inputs,
            batch_size=batch_size,
            batch_scheduler_buffer_size=batch_scheduler_buffer_size,
            adaptive_batch_size=adaptive_batch_size,
            progress_interval=progress_interval,
            force=force,
            cache_only=cache_only,
            verbose=verbose,
            log_level=log_level,
            total_num_inputs=total_num_inputs,
            **kwargs,
        )

    def run(  # type:ignore[override]
        self,
        texts: Iterable[str],
        truncate: bool = False,
        instruction: None | str = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        batch_scheduler_buffer_size: None | int = None,
        adaptive_batch_size: bool = True,
        progress_interval: None | int = 60,
        force: bool = False,
        cache_only: bool = False,
        verbose: None | bool = None,
        log_level: None | int = None,
        total_num_texts: None | int = None,
        return_generator: bool = False,
        **kwargs,
    ) -> Generator[np.ndarray, None, None] | list[np.ndarray]:
        assert (
            not _is_instructor_model(self.model_name) or instruction is not None
        ), "Instructor models require the `instruction` parameter."

        def get_max_length_function() -> dict[str, Any]:
            def max_length_func(texts: list[str]) -> int:
                if _is_instructor_model(self.model_name):  # pragma: no cover
                    return max(
                        [
                            self.count_tokens(cast(str, instruction) + " " + t)
                            for t in texts
                        ]
                    )
                else:
                    return max([self.count_tokens(t) for t in texts])

            return {"max_length_func": max_length_func}

        results_generator = self._run_over_batches(
            run_batch=self._run_batch,
            get_max_input_length_function=get_max_length_function,
            max_model_length=self.model_max_length,
            inputs=texts,
            truncate=truncate,
            instruction=instruction,
            batch_size=batch_size,
            batch_scheduler_buffer_size=batch_scheduler_buffer_size,
            adaptive_batch_size=adaptive_batch_size,
            progress_interval=progress_interval,
            force=force,
            cache_only=cache_only,
            verbose=verbose,
            log_level=log_level,
            total_num_inputs=total_num_texts,
            **kwargs,
        )
        if not return_generator:
            return list(results_generator)
        else:
            return results_generator

    @cached_property
    def model_card(self) -> None | str:
        return get_model_card_url(self.hf_model_name)

    @cached_property
    def license(self) -> None | str:
        return get_license_info(self.hf_model_name, repo_type="model", revision=None)

    @cached_property
    def citation(self) -> None | list[str]:
        model_citations = get_citation_info(
            self.hf_model_name, repo_type="model", revision=None
        )
        citations = []
        citations.append(
            """
@inproceedings{reimers-2019-sentence-bert,
  title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
  author = "Reimers, Nils and Gurevych, Iryna",
  booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural"""
            """ Language Processing",
  month = "11",
  year = "2019",
  publisher = "Association for Computational Linguistics",
  url = "https://arxiv.org/abs/1908.10084",
}
            """.strip()
        )
        if _is_instructor_model(self.model_name):  # pragma: no cover
            citations.append(
                """
@inproceedings{INSTRUCTOR,
  title={One Embedder, Any Task: Instruction-Finetuned Text Embeddings},
  author={Su, Hongjin and Shi, Weijia and Kasai, Jungo and Wang, Yizhong and Hu,"""
                """ Yushi and  Ostendorf, Mari and Yih, Wen-tau and Smith, Noah A. and"""
                """  Zettlemoyer, Luke and Yu, Tao},
  url={https://arxiv.org/abs/2212.09741},
  year={2022},
}
                """.strip()
            )
        if isinstance(model_citations, list):  # pragma: no cover
            citations.extend(model_citations)
        return citations

    @property
    def version(self) -> float:
        return 1.0

    @cached_property
    def display_name(self) -> str:
        return super().display_name + f" ({self.model_name})"

    @cached_property
    def _cache_name(self) -> None | str:
        names = [safe_fn(self.model_name, allow_slashes=False)]
        names.append(
            str(self.dtype)
            if self.dtype is not None
            else (str(torch.get_default_dtype()))
        )
        to_hash: list[Any] = []
        if len(to_hash) > 0:  # pragma: no cover
            names.append(Hasher.hash(to_hash))
        return "_".join(names)

    def unload_model(self):
        # Delete cached model and tokenizer
        if "model" in self.__dict__:
            del self.__dict__["model"]
        if "tokenizer" in self.__dict__:
            del self.__dict__["tokenizer"]

        # Garbage collect
        gc.collect()

        # Clear CUDA cache
        if torch.cuda.is_available():  # pragma: no cover
            torch.cuda.empty_cache()

    def __getstate__(self):  # pragma: no cover
        state = super().__getstate__()

        # Remove cached model or tokenizer before serializing
        state.pop("model", None)
        state.pop("tokenizer", None)

        return state


__all__ = ["SentenceTransformersEmbedder"]
