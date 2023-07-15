import warnings
from typing import Any

from dill import dumps, loads

from datasets.features.features import Features, Value

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


def unpickle_transform(batch, features=__FEATURES_DEFAULT):
    for column in batch:
        feature = features.get(column, None)
        if not isinstance(feature, Value) or feature.dtype != "binary":
            continue
        for i in range(len(batch[column])):
            v = batch[column][i]
            if (
                isinstance(v, bytes)
                and len(v) >= 2
                and v[0] == 128
                and v[-1] == 46
                and _PICKLE_KEY.encode("utf8") in v[:100]
            ):
                batch[column][i] = unpickle(v)
            else:
                batch[column][i] = v
    return batch
