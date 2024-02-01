import warnings
from typing import Any

from datasets.features.features import Features, Value
from dill import dumps, loads

_INTERNAL_PICKLE_KEY = "__DataDreamer__pickle_internal__"
_PICKLE_KEY = "__DataDreamer__pickle__"
__FEATURES_DEFAULT = Features()


def _pickle(value: Any, *args: Any, **kwargs: Any) -> bytes:
    if _INTERNAL_PICKLE_KEY not in kwargs:
        warnings.warn(
            "Do not call pickle() directly. You should instead use the .pickle()"
            " method on a Step object.",
            stacklevel=2,
        )
    else:
        del kwargs[_INTERNAL_PICKLE_KEY]
    return dumps({_PICKLE_KEY: value}, *args, **kwargs)


def unpickle(value: bytes) -> Any:
    return loads(value)[_PICKLE_KEY]


def _unpickle_transform_value(value):
    if (
        isinstance(value, bytes)
        and len(value) >= 2
        and value[0] == 128
        and value[-1] == 46
        and _PICKLE_KEY.encode("utf8") in value[:100]
    ):
        return unpickle(value)
    else:
        return value


def unpickle_transform(batch, features=__FEATURES_DEFAULT, batched=False):
    for column in batch:
        feature = features.get(column, None)
        if not isinstance(feature, Value) or feature.dtype != "binary":
            continue
        if batched:
            for i in range(len(batch[column])):
                batch[column][i] = _unpickle_transform_value(batch[column][i])
        else:
            batch[column] = _unpickle_transform_value(batch[column])
    return batch
