import re
import uuid
from typing import Any, cast

import dill
from datasets import Dataset
from datasets.fingerprint import Hasher
from dill.source import getsource

from .. import DataDreamer
from ..utils.hf_model_utils import get_base_model_from_peft_model, is_peft_model
from ..utils.import_utils import ignore_transformers_warnings

with ignore_transformers_warnings():
    from transformers import PreTrainedModel, PreTrainedTokenizerBase


class _DatasetGeneratorPickleHack:
    # Source: https://github.com/huggingface/datasets/issues/6194

    def __init__(self, generator, generator_id=None):
        self.generator = generator
        self.generator_is_picklable = dill.pickles(self.generator)
        self.generator_id = (
            generator_id if generator_id is not None else str(uuid.uuid4())
        )

    def __call__(self, *args, **kwargs):
        return self.generator(*kwargs, **kwargs)

    def __reduce__(self):  # pragma: no cover
        if not self.generator_is_picklable and not DataDreamer.is_background_process():
            return (_DatasetGeneratorPickleHack_raise, (self.generator_id,))
        elif not self.generator_is_picklable and DataDreamer.is_background_process():
            raise RuntimeError(
                "A non-picklable generator was returned from the step running in the"
                " background. Try setting `background=False`."
            )
        else:
            return super().__reduce__()


def _DatasetGeneratorPickleHack_raise(*args, **kwargs):  # pragma: no cover
    raise RuntimeError(
        "A non-picklable generator was returned from the step."
        " If you are using `background=True`, try setting `background=False`."
    )


def stable_fingerprint(value: Any) -> str:
    from ..trainers import Trainer

    if isinstance(value, (list, tuple, set)):
        return Hasher.hash([type(value), [stable_fingerprint(v) for v in value]])
    elif isinstance(value, dict):
        return Hasher.hash({k: stable_fingerprint(v) for k, v in value.items()})
    else:
        if isinstance(value, Trainer):
            assert value._done, f"Trainer '{value.name}' has not been run yet. Use `.train()` to start training."
            return cast(str, value.fingerprint)
        if is_peft_model(value):  # pragma: no cover
            return stable_fingerprint(
                [
                    value.__class__.__name__,
                    value.peft_config,
                    value.dtype,
                    value.state_dict(),
                    get_base_model_from_peft_model(value),
                ]
            )
        elif isinstance(value, PreTrainedModel):
            return stable_fingerprint(
                [
                    value.__class__.__name__,
                    value.config._name_or_path,
                    value.config,
                    value.dtype,
                    value.state_dict(),
                ]
            )
        elif isinstance(value, PreTrainedTokenizerBase):
            return stable_fingerprint([value.__class__.__name__, value.name_or_path])
        elif type(value) is type or callable(value):
            try:
                return Hasher.hash(
                    re.sub(r" at 0x[0-9a-f]+", " at 0x0", getsource(value))
                )
            except IOError:  # pragma: no cover
                return Hasher.hash(value)
        elif isinstance(value, Dataset):  # pragma: no cover
            return value._fingerprint  # type:ignore[attr-defined]
        else:
            return Hasher.hash(value)
