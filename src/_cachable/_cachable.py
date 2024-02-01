import itertools
import logging
import os
import tempfile
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from collections.abc import Iterator, Sized
from contextlib import nullcontext
from functools import cached_property, partial
from itertools import chain, islice, tee
from logging import Logger
from math import ceil
from time import time
from typing import Any, Callable, DefaultDict, Generator, Iterable, cast
from uuid import uuid4

import psutil
import torch
from datasets.fingerprint import Hasher
from filelock import FileLock, Timeout
from sortedcontainers import SortedDict
from sqlitedict import SqliteDict

from .. import DataDreamer, logging as datadreamer_logging
from ..logging import DATEFMT, logger as datadreamer_logger
from ..project.environment import RUNNING_IN_PYTEST
from ..utils.fingerprint_utils import stable_fingerprint
from ..utils.time_utils import progress_eta

ADAPTIVE_BATCH_SIZE_SAMPLES_CONFIDENCE_THRESHOLD = 10
ADAPTIVE_BATCH_SIZE_BINS = 6


def _is_primitive(value: Any) -> bool:
    return (
        isinstance(value, (int, float, str, bytes, bool, type(None)))
        or (
            isinstance(value, (list, tuple, set))
            and all(_is_primitive(v) for v in value)
        )
        or (isinstance(value, dict) and all(_is_primitive(v) for v in value.items()))
    )


def _default_batch_scheduler_buffer_size(batch_size: int) -> int:
    return max(batch_size * 10, 1000)


def _notify_adaptive_batch_sizing(model_logger: Logger, progress_state: dict[str, Any]):
    if model_logger.level > logging.DEBUG:
        if "notified_about_adaptive_batch_sizing" not in progress_state:
            model_logger.warning(
                "The batch size provided was too large (a memory error was triggered)."
                " Adaptive batch sizing will try to recover and learn what batch size"
                " is appropriate. Set `verbose=True` to see more logs."
            )
            progress_state["notified_about_adaptive_batch_sizing"] = True


class _StrWithSeed(str):
    seed: Any

    def __new__(cls, value: str, seed: "Any | _StrWithSeed"):
        obj = str.__new__(cls, value)
        obj.seed = seed.seed if isinstance(seed, _StrWithSeed) else seed
        return obj

    def __eq__(self, __value: object) -> bool:
        return (
            super().__eq__(__value)
            and isinstance(__value, _StrWithSeed)
            and self.seed == __value.seed
        )

    def __hash__(self):
        return hash((self.seed, str(self)))

    @staticmethod
    def total_per_input_seeds(inputs: list["str | _StrWithSeed"]) -> int:
        return sum(
            input.seed if isinstance(input, _StrWithSeed) else 0 for input in inputs
        )


class _Cachable(ABC):
    def __init__(self, cache_folder_path: None | str = None):
        self.adaptive_batch_sizes: DefaultDict[tuple, SortedDict] = defaultdict(
            SortedDict
        )
        self.cache_folder_path: None | str = cache_folder_path
        self._lock = self._create_lock()
        super().__init__()

    def _create_lock(self):
        cls_name = self.__class__.__name__
        return FileLock(
            os.path.join(tempfile.gettempdir(), f"{cls_name}-{uuid4().hex}.flock")
        )

    def __enter__(self):
        try:
            self._lock.acquire(timeout=0)
        except Timeout:
            raise RuntimeError(
                f"You may be trying to access {self.display_name} in two different"
                " threads/processes concurrently. Please create two separate"
                f" {self.display_name} objects if you wish to do that."
            ) from None

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._lock.release()

    def get_logger(
        self, key: str, verbose: None | bool = None, log_level: None | int = None
    ) -> Logger:
        stderr_handler = logging.StreamHandler()
        stderr_handler.setLevel(logging.NOTSET)
        cls_name = self.__class__.__name__
        logger = logging.getLogger(f"datadreamer.{cls_name}.{key}")
        if RUNNING_IN_PYTEST:
            logger.propagate = True
        else:
            logger.propagate = False  # pragma: no cover
        log_format: str = (
            datadreamer_logger.handlers[0].formatter
            and datadreamer_logger.handlers[0].formatter._fmt
        ) or datadreamer_logging.STANDARD_FORMAT
        log_format = log_format.replace(
            "%(message)s", f"[{self.display_icon}{self.display_name}] %(message)s"
        )
        formatter = logging.Formatter(log_format, datefmt=DATEFMT, validate=False)
        stderr_handler.setFormatter(formatter)
        logger.handlers.clear()
        logger.addHandler(stderr_handler)
        effective_level = datadreamer_logger.level if log_level is None else log_level
        if verbose:
            logger.setLevel((min(logging.DEBUG, effective_level)))
        elif verbose is False:
            logger.setLevel(logging.CRITICAL + 1)
        else:
            logger.setLevel(effective_level)
        return logger

    @cached_property
    def cache_and_lock(self) -> None | tuple[SqliteDict, FileLock]:
        cls_name = self.__class__.__name__
        cache_folder_path: str
        if self.cache_folder_path is not None:
            cache_folder_path = cast(str, self.cache_folder_path)
        elif DataDreamer.initialized() and not DataDreamer.is_running_in_memory():
            cache_folder_path = os.path.join(
                DataDreamer.get_output_folder_path(), ".cache"
            )
        else:
            return None
        db_fn = (
            f"{cls_name}_{self._cache_name}.db"
            if self._cache_name
            else f"{cls_name}.db"
        )
        db_path = os.path.join(cache_folder_path, db_fn)
        db_lock_path = db_path + ".flock"
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        cache_db_lock = FileLock(db_lock_path)
        with cache_db_lock:
            cache_db = SqliteDict(
                db_path, tablename="run_cache", journal_mode="WAL", autocommit=False
            )
        if self.cache_folder_path is None:
            if db_fn not in DataDreamer.ctx.caches:
                DataDreamer.ctx.caches[db_fn] = cache_db, cache_db_lock
            return DataDreamer.ctx.caches[db_fn]
        else:
            return cache_db, cache_db_lock

    def _compute_cache_key(self, args_cache_key: str, input: Any) -> str:
        per_input_seed = None
        if isinstance(input, _StrWithSeed):
            per_input_seed = input.seed
            input = str(input)
        to_hash = dict(args_cache_key=args_cache_key, input=input)
        if per_input_seed is not None:
            to_hash["per_input_seed"] = per_input_seed
        return Hasher.hash(to_hash)

    def _batch_cache_lookup(
        self, inputs: list[Any], input_to_cache_key: dict[Any, str]
    ) -> dict[Any, Any]:
        if self.cache_and_lock:
            cache, _ = cast(tuple[SqliteDict, FileLock], self.cache_and_lock)
            cache_results = {}
            inputs_iter = iter(
                # Unique-ify inputs
                dict([(stable_fingerprint(input), input) for input in inputs]).values()
            )
            while True:
                inputs_batch = list(islice(inputs_iter, 900))
                if len(inputs_batch) == 0:
                    break
                cache_query = (
                    "SELECT key, value FROM run_cache"
                    f' WHERE key IN ({",".join(["?"] * len(inputs_batch))})'
                )
                cache_results.update(
                    {
                        result[0]: cache.decode(result[1])
                        for result in cache.conn.select(
                            cache_query,
                            [
                                input_to_cache_key[stable_fingerprint(input)]
                                for input in inputs_batch
                            ],
                        )
                    }
                )
            return cache_results
        else:  # pragma: no cover
            return {}

    def _is_batch_size_exception(self, e: BaseException) -> bool:
        return isinstance(e, (MemoryError, OSError))

    def reset_adaptive_batch_sizing(self):
        self.adaptive_batch_sizes = defaultdict(SortedDict)

    def _adaptive_run_batch(  # noqa: C901
        self,
        run_batch: Callable[..., list[Any]],
        get_max_input_length_function: Callable[[], dict[str, Any]],
        max_model_length: int | Callable,
        inputs: list[Any],
        batch_size: int = 1,
        model_logger: None | Logger = None,
        progress_state: None | dict[str, Any] = None,
        **kwargs,
    ) -> list[Any]:
        assert model_logger is not None
        assert progress_state is not None

        # Initialize variables for adaptive batch sizing
        predicted_results_sub_batches = []
        last_idx = 0
        system_memory = psutil.virtual_memory().total
        cuda_memory = tuple(
            [
                torch.cuda.get_device_properties(d).total_memory
                for d in range(torch.cuda.device_count())
            ]
        )
        adaptive_batch_sizes = self.adaptive_batch_sizes[
            ((system_memory, cuda_memory), kwargs.get("max_new_tokens", None))
        ]
        get_max_input_length_function_results = get_max_input_length_function()
        max_length_func = get_max_input_length_function_results["max_length_func"]

        # Discretize the max_input_length into bins
        def binify(value: int) -> int:
            max_context_length = (
                cast(int, max_model_length(kwargs.get("max_new_tokens", None) or 0))
                if callable(max_model_length)
                else max_model_length
            )
            num_bins = min(ADAPTIVE_BATCH_SIZE_BINS, max_context_length)
            bin_idx = min(value // (max_context_length // num_bins), num_bins - 1)
            return (
                ((bin_idx + 1) * (max_context_length // num_bins))
                if bin_idx < (num_bins - 1)
                else (max_context_length + 1)
            )

        # While there are inputs left to process
        while last_idx < len(inputs):
            # Get the maximum input length in the batch
            max_input_length = binify(max_length_func(inputs))

            # Adaptive batch size estimation routine - Estimate what batch size will
            # work for this input length without throwing OOM errors by using data
            # from past runs.
            total = (
                sum(adaptive_batch_sizes[max_input_length].values())
                if max_input_length in adaptive_batch_sizes
                else None
            )
            if (
                max_input_length in adaptive_batch_sizes
                and total >= ADAPTIVE_BATCH_SIZE_SAMPLES_CONFIDENCE_THRESHOLD
            ):
                current_bz = adaptive_batch_sizes[max_input_length].most_common(1)[0][0]
            else:
                left_key_idx = adaptive_batch_sizes.bisect_left(max_input_length) - 1
                right_key_idx = adaptive_batch_sizes.bisect_right(max_input_length)
                if left_key_idx < 0 and right_key_idx >= len(
                    adaptive_batch_sizes.keys()
                ):
                    # We have no information yet, just use the full batch size
                    current_bz = batch_size
                elif left_key_idx >= 0 and right_key_idx >= len(
                    adaptive_batch_sizes.keys()
                ):
                    # We've seen shorter input lengths in the past, so we can use a
                    # successful batch size from them as a starting point
                    current_bz = min(
                        adaptive_batch_sizes[adaptive_batch_sizes.keys()[left_key_idx]]
                    )
                elif left_key_idx < 0 and right_key_idx < len(
                    adaptive_batch_sizes.keys()
                ):
                    # We've only seen longer input lengths in the past, which doesn't
                    # give us a good upper bound, so we just use the full batch size
                    current_bz = batch_size
                else:
                    # We've seen shorter and longer input lengths in the past, so we
                    # can use the average of successful batch sizes from them as a
                    # starting point
                    upper = min(
                        adaptive_batch_sizes[adaptive_batch_sizes.keys()[left_key_idx]]
                    )
                    lower = max(
                        adaptive_batch_sizes[adaptive_batch_sizes.keys()[right_key_idx]]
                    )
                    current_bz = (upper + lower) // 2

            # Make sure we never exceed the user's requested batch size
            current_bz = min(current_bz, batch_size)

            # Continuously try to run the model with a sub-batch of inputs, making the
            # batch smaller and smaller until it works
            for attempt in itertools.count(start=1):
                inputs_sub_batch = inputs[last_idx : last_idx + current_bz]
                try:
                    predicted_results_sub_batch = run_batch(
                        inputs=inputs_sub_batch,
                        batch_size=batch_size,
                        **kwargs,
                        **get_max_input_length_function_results,
                    )

                    # If it worked, then we save the results...
                    predicted_results_sub_batches.append(predicted_results_sub_batch)

                    # ...and we record what batch size worked for the current
                    # max input length, to help create better future estimates
                    if len(inputs_sub_batch) == current_bz:
                        max_input_length = binify(max_length_func(inputs_sub_batch))
                        if max_input_length not in adaptive_batch_sizes:
                            adaptive_batch_sizes[max_input_length] = Counter()
                        bz_counter = adaptive_batch_sizes[max_input_length]
                        if len(bz_counter) >= 2:
                            two_most_common_batch_sizes = adaptive_batch_sizes[
                                max_input_length
                            ].most_common(2)
                            if current_bz == two_most_common_batch_sizes[0][0]:
                                if (
                                    two_most_common_batch_sizes[0][1]
                                    - two_most_common_batch_sizes[1][1]
                                ) <= 10:
                                    # Never allow the most frequent batch size to outrun
                                    # the second most frequent batch size in count
                                    # to prevent it from being impossible for the second
                                    # most frequent batch size to catch up. This way,
                                    # the second most frequent batch size can always
                                    # catch up in around ~10 runs.
                                    bz_counter[current_bz] += 1
                            else:
                                bz_counter[current_bz] += 1
                        else:
                            bz_counter[current_bz] += 1

                        # Remove batch sizes that worked in the past,
                        # but are not frequently successful
                        if max_input_length in adaptive_batch_sizes:
                            total = sum(bz_counter.values())
                            if (
                                total
                                >= ADAPTIVE_BATCH_SIZE_SAMPLES_CONFIDENCE_THRESHOLD
                            ):
                                most_common_freq = (
                                    bz_counter.most_common(1)[0][1] / total
                                )
                                if most_common_freq > 0.3:
                                    for bz in list(bz_counter.keys()):
                                        freq = bz_counter[bz] / total
                                        # Remove infrequent batch sizes, but always
                                        # keep at least 3 batch sizes
                                        if freq < 0.1 and len(bz_counter) > 3:
                                            del bz_counter[bz]

                        # Log to the user
                        if attempt > 1:
                            _notify_adaptive_batch_sizing(
                                model_logger=model_logger, progress_state=progress_state
                            )
                            model_logger.debug(
                                "Adaptive batch sizing successfully recovered"
                                f" with batch size of {current_bz}."
                                " This will help select a better batch"
                                " size in the future after enough samples are"
                                " collected."
                            )

                    # ...and we update where we are in the lists of inputs
                    last_idx += len(inputs_sub_batch)

                    # ...finally, we exit the while-loop
                    break
                except (torch.cuda.OutOfMemoryError, Exception) as e:
                    if not isinstance(e, torch.cuda.OutOfMemoryError):
                        if not self._is_batch_size_exception(e):
                            raise

                    #####################################################
                    # Commenting these out temporarily for performance as
                    # memory-cleanup tends to be a slow operation
                    #####################################################
                    # # Garbage collect
                    # gc.collect()

                    # # Clear CUDA cache
                    # if torch.cuda.is_available():  # pragma: no cover
                    #     torch.cuda.empty_cache()

                    # Catch memory errors
                    if current_bz == 1:
                        # If we got a memory error on batch size 1, raise it, we cannot
                        # make the batch size any smaller
                        raise e
                    else:
                        # Otherwise, reduce the batch size by 10%
                        new_current_bz = max(1, int(current_bz * 0.9))

                        # Update the batch size
                        _notify_adaptive_batch_sizing(
                            model_logger=model_logger, progress_state=progress_state
                        )
                        model_logger.debug(
                            f"A batch size of {current_bz} failed."
                            " Adaptive batch sizing will try a"
                            f" smaller batch size of {new_current_bz}."
                        )
                        current_bz = new_current_bz

        # Combine all the results from all the sub-batches
        return list(chain.from_iterable(predicted_results_sub_batches))

    def _run_over_sorted_batches(  # noqa: C901
        self,
        run_batch: Callable[..., list[Any]],
        get_max_input_length_function: None | Callable[[], dict[str, Any]],
        max_model_length: None | int | Callable,
        inputs: Iterable[Any],
        batch_size: int = 1,
        adaptive_batch_size: bool = True,
        force: bool = False,
        cache_only: bool = False,
        model_logger: None | Logger = None,
        progress_state: None | dict[str, Any] = None,
        **kwargs,
    ) -> list[Any]:
        assert progress_state is not None
        run_args = dict(batch_size=batch_size, **kwargs)
        assert batch_size > 0, "Batch size must be greater than 0."

        # Hash generation arguments that will be used as part of the cache key
        args_cache_key = Hasher.hash(
            dict(
                _model_version=self.version,
                kwargs={
                    kwarg: value
                    for kwarg, value in kwargs.items()
                    if _is_primitive(value)
                },
            )
        )

        # Create an iterator over the inputs
        inputs_iter_1, inputs_iter_2, inputs_iter_3 = tee(iter(inputs), 3)

        # Look up these inputs in the cache
        input_to_cache_key = {}
        if self.cache_and_lock:
            cache, lock = cast(tuple[SqliteDict, FileLock], self.cache_and_lock)

            # Compute the cache key for all inputs
            input_to_cache_key.update(
                {
                    stable_fingerprint(input): self._compute_cache_key(
                        args_cache_key, input
                    )
                    for input in inputs_iter_1
                }
            )

            # Get the cached results for all inputs, if they exist in the
            # cache
            if not force:
                cache_results = self._batch_cache_lookup(
                    list(inputs_iter_2), input_to_cache_key
                )
            else:
                cache_results = {}
        else:
            cache_results = {}

        # While there are more batches of inputs to process, get the results
        predicted_results = []
        while True:
            # Now we find:
            # inputs_to_query_batch - a maximum of batch_size number of inputs
            # to run (that were not available in the cache)
            # predicted_results_batch - the results of cached inputs we can return
            # immediately + placeholder None's for the inputs_to_query_batch
            # that need to be run to get the results for
            remaining_to_query = batch_size  # We find a maximum of batch_size inputs
            predicted_results_batch: list[None | Any] = []
            inputs_to_query_batch = []

            # While we haven't found a maximum of batch_size number of inputs yet
            while remaining_to_query > 0:
                # Get a batch inputs
                inputs_batch = list(islice(inputs_iter_3, remaining_to_query))
                if len(inputs_batch) == 0:
                    # Exit if there are no more
                    break

                # For each input in the batch
                for input in inputs_batch:
                    cache_key = input_to_cache_key.get(stable_fingerprint(input), None)
                    if cache_key in cache_results:
                        # If they exist in the cache, we can add the result directly
                        predicted_results_batch.append(cache_results[cache_key])
                    else:
                        # If they don't exist in the cache, we need to run this input
                        inputs_to_query_batch.append(input)
                        # And we add a placeholder None value that will get replaced
                        # later
                        predicted_results_batch.append(None)

                # We found len(inputs_to_query_batch) inputs to query, so we need to
                # now only find remaining_to_query - len(inputs_to_query_batch) more.
                remaining_to_query -= len(inputs_to_query_batch)

            # Log progress
            progress_state["log_progress"](post_update=len(predicted_results_batch))

            # Exit if there are no more results to add
            if len(predicted_results_batch) == 0:
                break

            # Run the inputs that didn't exist in the cache
            results_iter: Iterator[Any]
            if len(inputs_to_query_batch) > 0:
                if cache_only:
                    raise RuntimeError(
                        "Requested cache_only=True, but some inputs were not found"
                        " in the cache and must be run against the model."
                    )
                if adaptive_batch_size:
                    if len(self.adaptive_batch_sizes) == 0 and self.cache_and_lock:
                        cache, _ = cast(
                            tuple[SqliteDict, FileLock], self.cache_and_lock
                        )
                        if "adaptive_batch_sizes" in cache:
                            self.adaptive_batch_sizes = cache["adaptive_batch_sizes"]
                    assert get_max_input_length_function is not None
                    assert max_model_length is not None
                    results_iter = iter(
                        self._adaptive_run_batch(
                            run_batch=run_batch,
                            get_max_input_length_function=get_max_input_length_function,
                            max_model_length=max_model_length,
                            inputs=inputs_to_query_batch,
                            model_logger=model_logger,
                            progress_state=progress_state,
                            **run_args,
                        )
                    )
                else:
                    if get_max_input_length_function is None:
                        get_max_input_length_function_results = {}
                    else:
                        get_max_input_length_function_results = (
                            get_max_input_length_function()
                        )
                    results_iter = iter(
                        run_batch(
                            inputs=inputs_to_query_batch,
                            **run_args,
                            **get_max_input_length_function_results,
                        )
                    )
            else:
                results_iter = iter([])

            # Fill in the placeholder None's with the results of running the inputs
            if self.cache_and_lock:
                inputs_to_query_batch_iter = iter(inputs_to_query_batch)
                cache, lock = cast(tuple[SqliteDict, FileLock], self.cache_and_lock)
                with lock:
                    cache["adaptive_batch_sizes"] = self.adaptive_batch_sizes
                    for idx, predicted_result in enumerate(predicted_results_batch):
                        # Find the placeholder None's
                        if predicted_result is None:
                            # Get the next input and result
                            input = next(inputs_to_query_batch_iter)
                            result = next(results_iter)

                            # Store the value in the cache for the future
                            cache_key = input_to_cache_key[stable_fingerprint(input)]
                            cache[cache_key] = result

                            # Fill in the placeholder None
                            predicted_results_batch[idx] = result
                    cache.commit()
            else:
                for idx, predicted_result in enumerate(predicted_results_batch):
                    # Find the placeholder None's
                    if predicted_result is None:
                        # Get the next result
                        result = next(results_iter)

                        # Fill in the placeholder None
                        predicted_results_batch[idx] = result

            # Add the results of the batch of inputs to the total results
            predicted_results.extend(predicted_results_batch)

        # Return all results
        return cast(list[Any], predicted_results)

    def _run_over_batches_locked(  # noqa: C901
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
        # Create a progress tracker
        def log_progress(
            model_logger, progress_interval, progress_state, pre_update=0, post_update=0
        ):
            # Log progress
            progress_state["progress_inputs"] += pre_update
            if (
                progress_interval is not None
                and (time() - progress_state["progress_last"]) > progress_interval
            ):
                progress_state["progress_last"] = time()
                if progress_state["total_num_inputs"] is None:
                    if progress_state["step"]:
                        progress_state["step"]._set_progress_rows(
                            progress_state["progress_inputs"]
                        )
                    else:
                        model_logger.info(
                            f"Progress:"
                            f" {progress_state['progress_inputs']} {self._input_type}(s) 🔄"
                        )
                else:
                    progress_float = (
                        progress_state["progress_inputs"]
                        / progress_state["total_num_inputs"]
                    )
                    progress_int = min(100, max(0, int((progress_float) * 100)))
                    if progress_state["step"]:
                        progress_state["step"].progress = progress_float
                    else:
                        eta = progress_eta(
                            progress_float, progress_state["progress_start"]
                        )
                        model_logger.info(f"Progress: {progress_int}% 🔄 {eta}")
            progress_state["progress_inputs"] += post_update

        step = kwargs.pop("_step", None)
        step_progress = kwargs.pop("_step_progress", True)
        step_log_level = step.logger.level if step is not None else None
        model_logger = self.get_logger(
            key="run", verbose=verbose, log_level=log_level or step_log_level
        )
        progress_state: dict[str, Any] = kwargs.pop("progress_state", {})
        progress_state["progress_start"] = progress_state.get("progress_start", time())
        progress_state["progress_last"] = progress_state.get("progress_last", time())
        progress_state["progress_inputs"] = progress_state.get("progress_inputs", 0)
        progress_state["total_num_inputs"] = progress_state.get(
            "total_num_inputs",
            len(inputs) if isinstance(inputs, Sized) else total_num_inputs,
        )
        progress_state["log_progress"] = partial(
            log_progress,
            model_logger=model_logger,
            progress_interval=progress_interval,
            progress_state=progress_state,
        )
        progress_state["step"] = step if step_progress else None

        # Gather arguments to run the model
        run_args = dict(
            run_batch=run_batch,
            get_max_input_length_function=get_max_input_length_function,
            max_model_length=max_model_length,
            batch_size=batch_size,
            adaptive_batch_size=adaptive_batch_size,
            force=force,
            cache_only=cache_only,
            model_logger=model_logger,
            progress_state=progress_state,
            **kwargs,
        )

        # Select a proper batch_scheduler_buffer_size (to the nearest multiple of
        # batch_size)
        should_sort = True
        if batch_scheduler_buffer_size is not None and batch_scheduler_buffer_size < 0:
            should_sort = False
            batch_scheduler_buffer_size = None
        if batch_scheduler_buffer_size is None:
            batch_scheduler_buffer_size = _default_batch_scheduler_buffer_size(
                batch_size
            )
        batch_scheduler_buffer_size = max(batch_size, batch_scheduler_buffer_size)
        batch_scheduler_buffer_size = (
            batch_scheduler_buffer_size
            if batch_size == 0
            else ceil(batch_scheduler_buffer_size / batch_size) * batch_size
        )

        # Create an iterator over the inputs
        inputs_iter = iter(inputs)

        # Figure out how to re-order the inputs in a "scheduled" order
        while True:
            # Get a sorted batch of inputs (shortest to longest in length)
            # to minimize sending "jagged" tensors (padding) to the model that
            # wastes processing power
            inputs_batch = list(islice(inputs_iter, batch_scheduler_buffer_size))
            if len(inputs_batch) == 0:
                # Exit if there are no more
                break
            if should_sort:
                sorted_input_idxs = sorted(
                    range(len(inputs_batch)),
                    key=lambda idx: len(inputs_batch[idx])
                    if isinstance(inputs_batch[idx], Sized)
                    else 0,
                )
                sorted_inputs_batch = sorted(
                    inputs_batch,
                    key=lambda input: len(input) if isinstance(input, Sized) else 0,
                )
            else:
                sorted_inputs_batch = inputs_batch

            # Get the results and restore the original order
            results = self._run_over_sorted_batches(
                inputs=sorted_inputs_batch, **run_args
            )
            predicted_results_batch: Any
            if should_sort:
                predicted_results_batch = [None] * len(sorted_inputs_batch)
                for input_idx, result in zip(sorted_input_idxs, results):
                    predicted_results_batch[input_idx] = result
            else:
                predicted_results_batch = results

            # Yield the results of the sorted batch of inputs
            yield from predicted_results_batch

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
        _should_lock = kwargs.pop("_should_lock", True)
        # https://github.com/python/mypy/issues/10109
        with self if _should_lock else nullcontext():  # type:ignore[attr-defined]
            yield from self._run_over_batches_locked(
                run_batch=run_batch,
                get_max_input_length_function=get_max_input_length_function,
                max_model_length=max_model_length,
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

    @cached_property
    def model_card(self) -> None | str:  # pragma: no cover
        return None

    @cached_property
    def license(self) -> None | str:  # pragma: no cover
        return None

    @cached_property
    def citation(self) -> None | list[str]:  # pragma: no cover
        return None

    @property
    def version(self) -> float:  # pragma: no cover
        return 1.0

    @cached_property
    def display_icon(self) -> str:  # pragma: no cover
        return ""

    @cached_property
    def display_name(self) -> str:
        cls_name = self.__class__.__name__
        return cls_name

    @cached_property
    def _cache_name(self) -> None | str:  # pragma: no cover
        return None

    @property
    @abstractmethod
    def _input_type(self) -> str:
        pass

    def __repr__(self) -> str:
        return f"<{self.display_name}>"

    def __getstate__(self):
        state = self.__dict__.copy()

        # Remove any locks
        del state["_lock"]
        state.pop("cache_and_lock", None)

        # Delete values cached via @ring
        for key in list(state.keys()):  # pragma: no cover
            if key.startswith("__wire"):
                del state[key]

        return state

    def __setstate__(self, state):  # pragma: no cover
        self.__dict__.update(state)

        # Restore lock
        self._lock = self._create_lock()


__all__ = ["_Cachable", "_default_batch_scheduler_buffer_size"]
