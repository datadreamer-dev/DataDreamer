from collections.abc import Generator, Iterable
from typing import Any, Callable, Optional, Union

from datasets import Dataset, IterableDataset


class Step:
    def __init__(
        self,
        name: str,
        input: Optional[
            Union[
                Dataset,
                IterableDataset,
            ]
        ],
        outputs: Union[str, list[str], tuple[str, ...]],
    ):
        self._name: str = name
        self.__running: bool = False
        self.__progress: float = None
        self.input = input
        self.__output: Union[Dataset, IterableDataset] = None
        if isinstance(outputs, list) and len(outputs) == 0:
            raise ValueError("The step must name its outputs")
        self.output_names: tuple[str, ...] = (
            tuple(outputs) if isinstance(outputs, list) else (outputs,)
        )

    @property
    def progress(self) -> Optional[float]:
        return self.__progress

    @progress.setter
    def progress(self, value: float):
        if value < 1.0:
            self.__running = True
        self.__progress = max(min(value, 1.0), 0.0)

    def __get_progress_string(self):
        if self.__progress is not None:
            progress_int = int(self.__progress * 100)
            return f"{progress_int}%"
        else:
            return "0%"

    @property
    def output(self) -> Union[Dataset, IterableDataset]:
        if self.__output is None:
            if self.__running and self.__progress is None:
                raise AttributeError(
                    "Step is still running. Output is not available yet."
                )
            elif self.__progress is None:
                raise AttributeError("Step has not been run. Output is not available.")
            else:
                raise AttributeError(
                    f"Step is still running ({self.__get_progress_string()})."
                    " Output is not available yet."
                )
        else:
            return self.__output

    # Generator function of batches

    @output.setter
    def output(  # noqa: C901
        self,
        value: Union[
            Dataset,
            IterableDataset,
            dict[str, Iterable[Any]],
            list[Any],
            tuple[Iterable[Any], ...],
            Callable[
                [],
                Generator[
                    dict[str, Iterable[Any], list[Any], tuple[Any, ...]], None, None
                ],
            ],
        ],
    ):
        # Create a Dataset if given a list or tuple of Datasets
        # or create an IterableDataset if given a list or tuple of IterableDatasets
        if type(value) in [list, tuple] and all(
            [isinstance(d, Dataset) for d in value]
        ):
            value = Dataset.zip(*value)
        elif type(value) in [list, tuple] and all(
            [isinstance(d, Dataset) or isinstance(d, IterableDataset) for d in value]
        ):
            value = Dataset.izip(*value)

        # Create a Dataset/generator function if given a dict
        if isinstance(value, dict) and set(self.output_names) == set(value.keys()):
            if any([callable(v) for v in value.values()]):
                # One of the values of the dictionary is a generator function,
                # create a generator function of dicts
                iters = [
                    (value[k]() if callable(value[k]) else iter(value[k]))
                    for k in self.output_names
                ]
                rows = zip(*iters)

                def to_dict_generator_wrapper():
                    for row in rows:
                        yield {k: v for k, v in zip(self.output_names, row)}

                value = to_dict_generator_wrapper
            else:
                value = Dataset.from_dict({k: value[k] for k in self.output_names})
        elif isinstance(value, dict):
            raise AttributeError(
                f"Expected {self.output_names} dict keys instead of {list(value.keys())}."
            )

        # If given a single list when more than one output force it into a tuple
        if (
            isinstance(value, list)
            and len(self.output_names) > 1
            and len(value) == len(self.output_names)
            and [isinstance(v, Iterable) for v in value]
        ):
            value = tuple(value)
        elif (
            isinstance(value, list)
            and len(self.output_names) > 1
            and type(value[0]) in [list, tuple]
            and len(value[0]) == len(self.output_names)
        ):
            value = tuple(zip(*value))

        # If given a single list
        if isinstance(value, list) and len(self.output_names) > 1:
            raise AttributeError(
                f"Expected {len(self.output_names)} outputs ({self.output_names})"
            )

        # If given a tuple with the wrong number of elements
        if isinstance(value, tuple) and len(self.output_names) != len(value):
            raise AttributeError(
                f"Expected {len(self.output_names)} outputs ({self.output_names})"
            )

        # Create a generator function if given a tuple with a generator function
        if isinstance(value, tuple) and any([callable(v) for v in value]):
            iters = [(v() if callable(v) else iter(v)) for v in value]
            rows = zip(*iters)

            def to_dict_generator_wrapper():
                for row in rows:
                    yield {k: v for k, v in zip(self.output_names, row)}

            value = to_dict_generator_wrapper

        # If given a Dataset with the wrong number of
        if (isinstance(value, Dataset) or isinstance(value, IterableDataset)) and set(
            value.column_names
        ) != set(self.output_names):
            raise AttributeError(
                f"Expected {self.output_names} columns instead of {value.column_names}."
            )

        # Create an IterableDataset if given a generator function of dicts
        if callable(value):
            # Make sure the generator returns a dict and the keys are correct
            try:
                first_row = next(value())
                if isinstance(first_row, dict) and set(self.output_names) != set(
                    value.keys()
                ):
                    raise AttributeError(
                        f"Expected {self.output_names} dict keys from generator"
                        f" function instead of {list(first_row.keys())}."
                    )
                elif type(first_row) in [list, tuple]:
                    if len(first_row) != len(self.output_names):
                        raise AttributeError(
                            f"Expected {len(self.output_names)} outputs"
                            " ({self.output_names}) from generator function"
                        )

                    def to_dict_generator_wrapper():
                        for row in value():
                            yield {k: v for k, v in zip(self.output_names, row)}

                    value = to_dict_generator_wrapper
            except StopIteration:
                pass

            # If so, convert the generator to an IterableDataset
            value = IterableDataset.from_generator(value)

        if isinstance(value, Dataset) or isinstance(value, IterableDataset):
            self.__output = value
        elif isinstance(value, tuple):
            self.__output = Dataset.from_dict(
                {k: v for k, v in zip(self.output_names, value)}
            )
        elif isinstance(value, list):
            self.__output = Dataset.from_dict({self.output_names[0]: value})
        self.__progress = 1.0
        self.__running = False
