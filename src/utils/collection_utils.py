import re
from typing import Any, Iterable


def uniq_str(collection: Iterable) -> str:
    seen = set()
    uniqed = tuple([x for x in collection if not (x in seen or seen.add(x))])  # type: ignore[func-returns-value]
    return re.sub(r",}$", "}", f"{{{str(uniqed)[1:-1]}}}") if uniqed else "{}"


def sort_keys(
    d: dict[Any, Any], key_order: list[Any]
) -> dict[Any, Any]:  # pragma: no cover
    d_copy = dict(d)
    all_keys = set(d_copy.keys())
    other_keys = all_keys.difference(set(key_order))
    d.clear()
    for key in key_order:
        if key in d_copy:
            d[key] = d_copy[key]
    for key in other_keys:
        d[key] = d_copy[key]
    return d
