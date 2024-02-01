import os

import ring


def lru(*args, **kwargs):
    if "SPHINX_BUILD" in os.environ:  # pragma: no cover

        def noop_decorator(func):
            return func

        return noop_decorator
    else:
        return ring.lru(*args, **kwargs)
