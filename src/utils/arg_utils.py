from typing import TypeVar


class Default:
    def __init__(self, name: str):
        self.name = name

    def __repr__(self):  # pragma: no cover
        return self.name


DEFAULT = Default("DEFAULT")
AUTO = Default("AUTO")

T = TypeVar("T")


def default_to(val: T | Default, default_val: T) -> T:
    if isinstance(val, Default):
        return default_val
    else:
        return val
