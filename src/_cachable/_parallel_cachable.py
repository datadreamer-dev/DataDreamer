import re
from concurrent.futures import ThreadPoolExecutor
from functools import cached_property, partial
from itertools import chain, islice
from time import time
from typing import Any, Generator, Iterable, Sized, Type

from filelock import Timeout

from .._cachable import _default_batch_scheduler_buffer_size
from ._cachable import _Cachable


class _ParallelCachable(_Cachable):
    def __init__(self, *cachables: _Cachable, cls: Type):
        _Cachable.__init__(self)

        # Get cachables
        self.cachables = list(cachables)

        # Validate cachables
        parallel_cls_name = self.__class__.__name__
        cls_name = cls.__name__
        assert len(self.cachables) > 0, f"You must pass at least 1 {cls_name}"
        assert all(
            isinstance(c, cls) for c in self.cachables
        ), f"{parallel_cls_name}() only {cls_name}s in the constructor."
        if len(set([(c.version, c._cache_name) for c in self.cachables])) > 1:
            raise ValueError(
                f"All {cls_name}s passed to {parallel_cls_name}() must be of the same type."
            )
        if len(set(self.cachables)) != len(self.cachables):
            raise ValueError(
                f"The same {cls_name} object was passed to {parallel_cls_name}()"
                " multiple times."
            )

        # Create thread pool
        self.pool = ThreadPoolExecutor(max_workers=len(self.cachables))

    def __enter__(self):
        for cachable in self.cachables:
            try:
                cachable._lock.acquire(timeout=0)
            except Timeout:
                raise RuntimeError(
                    f"You may be trying to access {self.display_name} in two different"
                    " threads/processes concurrently. Please create two separate"
                    f" {self.display_name} objects if you wish to do that."
                ) from None

    def __exit__(self, exc_type, exc_val, exc_tb):
        for cachable in self.cachables:
            cachable._lock.release()

    def reset_adaptive_batch_sizing(self):
        for cachable in self.cachables:
            cachable.reset_adaptive_batch_sizing()

    def _run_in_parallel(
        self, inputs: Iterable[Any], *args, **kwargs
    ) -> Generator[Any, None, None]:
        with self:
            batch_size: int = kwargs["batch_size"]
            batch_scheduler_buffer_size: int
            if (
                kwargs.get("batch_scheduler_buffer_size", None) is None
                or kwargs["batch_scheduler_buffer_size"] < 0
            ):
                batch_scheduler_buffer_size = _default_batch_scheduler_buffer_size(
                    batch_size
                )
            else:
                batch_scheduler_buffer_size = kwargs["batch_scheduler_buffer_size"]

            total_num_inputs: None | int = kwargs.get(
                f"total_num_{re.sub(r'y$', 'ie', self.cachables[0]._input_type)}s", None
            )

            # Iterate over batches
            progress_state: dict[str, Any] = {
                "progress_start": time(),
                "total_num_inputs": (
                    len(inputs) if isinstance(inputs, Sized) else total_num_inputs
                ),
            }
            inputs_iter = iter(inputs)
            while True:
                # Get number of models
                num_models = len(self.cachables)

                # Get a batch of inputs that is large enough to split across the models
                inputs_batch = list(
                    islice(inputs_iter, batch_scheduler_buffer_size * num_models)
                )
                if len(inputs_batch) == 0:
                    # Exit if there are no more
                    break

                # Sort the inputs by length
                sorted_prompt_idxs = sorted(
                    range(len(inputs_batch)), key=lambda idx: len(inputs_batch[idx])
                )
                sorted_inputs_batch = sorted(
                    inputs_batch, key=lambda prompt: len(prompt)
                )

                # Split the sorted inputs into num_models chunks, so each model gets
                # approximately similarly lengthed inputs
                sorted_prompt_idxs_per_models = chain.from_iterable(
                    [sorted_prompt_idxs[i::num_models] for i in range(num_models)]
                )
                sorted_inputs_batch_per_models = [
                    sorted_inputs_batch[i::num_models] for i in range(num_models)
                ]

                # Run the models in their own thread in parallel, get the results, and
                # restore the original order
                def _run_model(cachables, sorted_inputs_batch_per_models, arg):
                    model_idx, args, kwargs = arg
                    kwargs = kwargs.copy()
                    for kwarg in ["return_generator", "progress_state"]:
                        if kwarg in kwargs:
                            del kwargs[kwarg]
                    # Without list() below this wasn't properly running in parallel,
                    # should be fine to use memory-wise though, because this is still only
                    # materializing a batch, not all inputs
                    return list(
                        cachables[model_idx].run(
                            sorted_inputs_batch_per_models[model_idx],
                            *args,
                            **kwargs,
                            return_generator=True,
                            progress_state=progress_state,
                            _should_lock=False,
                        )
                    )

                run_model = partial(
                    _run_model, self.cachables, sorted_inputs_batch_per_models
                )

                predicted_results_batch: Any = [None] * len(sorted_inputs_batch)
                results = chain.from_iterable(
                    self.pool.map(
                        run_model,
                        [(model_idx, args, kwargs) for model_idx in range(num_models)],
                    )
                )
                for prompt_idx, result in zip(sorted_prompt_idxs_per_models, results):
                    predicted_results_batch[prompt_idx] = result

                # Yield the results run against each of the models
                yield from predicted_results_batch

    @cached_property
    def model_card(self) -> None | str:
        return self.cachables[0].model_card

    @cached_property
    def license(self) -> None | str:
        return self.cachables[0].license

    @cached_property
    def citation(self) -> None | list[str]:
        return self.cachables[0].citation

    @property
    def version(self) -> float:
        return self.cachables[0].version

    @cached_property
    def display_name(self) -> str:
        return self.cachables[0].display_name

    @cached_property
    def _cache_name(self) -> None | str:  # pragma: no cover
        return self.cachables[0]._cache_name

    def __getattr__(self, name):  # pragma: no cover
        return getattr(self.cachables[0], name)

    def __getstate__(self):  # pragma: no cover
        state = super().__getstate__()

        # Remove pool before serializing
        state["pool"] = None

        return state

    def __setstate__(self, state):  # pragma: no cover
        super().__setstate__(state)

        # Restore pool
        self.pool = ThreadPoolExecutor(max_workers=len(self.cachables))


__all__ = ["_ParallelCachable"]
