from functools import cache, cached_property, partial
from logging import root
from typing import Any, Callable

import torch
import torch._dynamo
from datasets.fingerprint import Hasher

from .. import DataDreamer
from ..logging import logger as datadreamer_logger
from ..utils.arg_utils import AUTO, Default
from ..utils.background_utils import RunIfTimeout
from ..utils.fs_utils import safe_fn
from ..utils.import_utils import ignore_hivemind_warnings, ignore_transformers_warnings
from .hf_transformers import HFTransformers

with ignore_transformers_warnings():
    from transformers import PreTrainedModel

_ServerInferenceSession_step: None | Callable = None


def _is_batch_size_exception_func(e: BaseException) -> bool:
    from hivemind.p2p.p2p_daemon_bindings.utils import P2PHandlerError

    return (
        isinstance(e, P2PHandlerError)
        and hasattr(e, "args")
        and isinstance(e.args, tuple)
        and len(e.args) > 0
        and "Could not allocate" in e.args[0]
        and "out of memory" in e.args[0]
    )


def _catch_memory_error__ServerInferenceSession_step(
    *args, **kwargs
):  # pragma: no cover
    import inspect

    # This monkey patch allows batch size errors to propagate properly back up to us
    def _patched_on_request_failure(
        _sequence_manager, old_on_request_failure, e, *args, **kwargs
    ):  # pragma: no cover
        # Restore original unpatched version of this function
        _sequence_manager.on_request_failure = old_on_request_failure
        raise e

    try:
        assert _ServerInferenceSession_step is not None
        return _ServerInferenceSession_step(*args, **kwargs)
    except Exception as e:  # pragma: no cover
        if _is_batch_size_exception_func(e):
            frame = inspect.currentframe()
            try:
                inference_session = frame.f_back.f_locals["self"]  # type: ignore[union-attr]
                inference_session._sequence_manager.on_request_failure = partial(
                    _patched_on_request_failure,
                    inference_session._sequence_manager,
                    inference_session._sequence_manager.on_request_failure,
                    e,
                )
            finally:
                del frame
        raise


@cache
def _monkey_patch_ServerInferenceSession_step():
    from hivemind.p2p.p2p_daemon_bindings.utils import P2PHandlerError  # noqa: F401

    try:
        _root_logger_handlers = root.handlers.copy()
        from petals.client.inference_session import _ServerInferenceSession
    finally:
        # Petals overrides the root logger, it should not do this, and we restore it
        # after importing Petals
        root.handlers = _root_logger_handlers

    _ServerInferenceSession.step = _catch_memory_error__ServerInferenceSession_step
    if DataDreamer.initialized():
        DataDreamer.ctx._monkey_patched_ServerInferenceSession_step = True


class Petals(HFTransformers):  # pragma: no cover
    def __init__(
        self,
        model_name: str,
        chat_prompt_template: None | str | Default = AUTO,
        system_prompt: None | str | Default = AUTO,
        revision: None | str = None,
        trust_remote_code: bool = False,
        device: None | int | str | torch.device = None,
        dtype: None | str | torch.dtype = None,
        adapter_name: None | str = None,
        cache_folder_path: None | str = None,
        **kwargs,
    ):
        global _ServerInferenceSession_step
        try:
            _root_logger_handlers = root.handlers.copy()
            from petals.client.inference_session import _ServerInferenceSession
        finally:
            # Petals overrides the root logger, it should not do this, and we restore it
            # after importing Petals
            root.handlers = _root_logger_handlers

        super().__init__(
            model_name=model_name,
            chat_prompt_template=chat_prompt_template,
            system_prompt=system_prompt,
            revision=revision,
            trust_remote_code=trust_remote_code,
            device=device,
            dtype=dtype,
            adapter_name=adapter_name,
            cache_folder_path=cache_folder_path,
            **kwargs,
        )
        if _ServerInferenceSession_step is None:
            _ServerInferenceSession_step = _ServerInferenceSession.step
        _monkey_patch_ServerInferenceSession_step()

    @cached_property
    def model(self) -> PreTrainedModel:
        try:
            _root_logger_handlers = root.handlers.copy()
            from petals import AutoDistributedModelForCausalLM
        finally:
            # Petals overrides the root logger, it should not do this, and we restore it
            # after importing Petals
            root.handlers = _root_logger_handlers

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
        with ignore_hivemind_warnings():
            model = AutoDistributedModelForCausalLM.from_pretrained(
                self.model_name,
                revision=self.revision,
                trust_remote_code=self.trust_remote_code,
                torch_dtype=self.dtype,
                active_adapter=self.adapter_name,
                **self.kwargs,
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

    def _is_batch_size_exception(self, e: BaseException) -> bool:
        return _is_batch_size_exception_func(e)

    def _run_batch(self, *args, **kwargs) -> list[str] | list[list[str]]:
        assert (
            kwargs.get("seed", None) is None
        ), f"`seed` is not supported for {type(self).__name__}"

        with self.model.inference_session(
            max_length=self.get_max_context_length(max_new_tokens=0)
        ) as sess:
            sess._sequence_manager.config.show_route = False
            kwargs["session"] = sess
            with ignore_hivemind_warnings():
                return super()._run_batch(*args, **kwargs)

    @cached_property
    def citation(self) -> None | list[str]:
        citations = super().citation or []
        citations.append(
            """
@article{borzunov2022petals,
  title = {Petals: Collaborative Inference and Fine-tuning of Large Models},
  author = {Borzunov, Alexander and Baranchuk, Dmitry and Dettmers,"""
            """ Tim and Ryabinin, Max and Belkada, Younes and Chumachenko,"""
            """ Artem and Samygin, Pavel and Raffel, Colin},
  journal = {arXiv preprint arXiv:2209.01188},
  year = {2022},
  url = {https://arxiv.org/abs/2209.01188}
}
        """.strip()
        )
        return citations

    @property
    def version(self) -> float:
        return 1.0

    @cached_property
    def _cache_name(self) -> None | str:
        names = [safe_fn(self.model_name, allow_slashes=False)]
        if self.adapter_name:
            names.append(safe_fn(self.adapter_name, allow_slashes=False))
        if self.revision:
            names.append(self.revision)
        names.append(
            str(self.dtype) if self.dtype is not None else str(self.model.dtype)
        )
        to_hash: list[Any] = []
        if len(to_hash) > 0:  # pragma: no cover
            names.append(Hasher.hash(to_hash))
        return "_".join(names)


__all__ = ["Petals"]
