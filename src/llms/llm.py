import os
import warnings
from abc import ABC, abstractmethod
from itertools import chain, islice

from filelock import FileLock
from sqlitedict import SqliteDict

from datasets.fingerprint import Hasher

from .. import DataDreamer


class LLM(ABC):
    def __init__(self):
        super().__init__()

    def get_cache(self) -> None | tuple[SqliteDict, FileLock]:
        cls_name = self.__class__.__name__
        if DataDreamer.initialized():
            if cls_name not in DataDreamer.ctx.llm_caches:
                db_path = os.path.join(
                    DataDreamer.get_output_folder_path(),
                    ".llm_cache",
                    (
                        f"{cls_name}_{self._cache_name}.db"
                        if self._cache_name
                        else f"{cls_name}.db"
                    ),
                )
                db_lock_path = db_path + ".flock"
                os.makedirs(os.path.dirname(db_path), exist_ok=True)
                cache_db = SqliteDict(
                    db_path, tablename="run_cache", journal_mode="WAL", autocommit=False
                )
                cache_db_lock = FileLock(db_lock_path)
                DataDreamer.ctx.llm_caches[cls_name] = cache_db, cache_db_lock
            return DataDreamer.ctx.llm_caches[cls_name]
        return None

    def _compute_cache_key(args_cache_key: str, prompt: str):
        return Hasher.hash(
            dict(
                args_cache_key=args_cache_key,
                prompt=prompt,
            )
        )

    @abstractmethod
    def count_tokens(self, value: str) -> int:
        pass

    def final_count_tokens(self, value: None | str) -> int:
        if value == "" or value is None:
            return 0
        else:
            return self.count_tokens(value)

    @abstractmethod
    def get_max_context_length(self, max_new_tokens: int) -> int:
        pass

    def get_prompt(  # noqa: C901
        self,
        max_new_tokens: int,
        beg_instruction: None | str = None,
        in_context_examples: None | list[str] = None,
        end_instruction: None | str = None,
        sep="\n",
    ) -> str:
        # Get the max context length
        max_context_length = self.get_max_context_length(max_new_tokens)

        # Initialize in_context_examples
        if in_context_examples is None:
            in_context_examples = []
        else:
            in_context_examples = in_context_examples.copy()
        assert in_context_examples is not None
        provided_in_context_examples_length = len(in_context_examples)

        # Get token counts of of beg_instruction, end_instruction, and sep
        beg_instruction_token_count = self.final_count_tokens(beg_instruction)
        end_instruction_token_count = self.final_count_tokens(end_instruction)
        sep_token_count = self.final_count_tokens(sep)

        # Get the minimum required token count for instructions
        required_token_count = 0
        if beg_instruction is not None:
            required_token_count += beg_instruction_token_count
            if len(in_context_examples) > 0:
                required_token_count += sep_token_count
            elif len(in_context_examples) == 0 and end_instruction is not None:
                required_token_count += sep_token_count
        if end_instruction is not None:
            required_token_count += end_instruction_token_count
            if len(in_context_examples) > 0:
                required_token_count += sep_token_count

        # Get how many tokens are left for in-context examples
        remaining_token_count = max_context_length - required_token_count
        if provided_in_context_examples_length > 0 and remaining_token_count <= 0:
            raise ValueError(
                "The prompt's template exceeds the max context length of the model"
                " and there is no room for in-context examples."
            )
        elif remaining_token_count < 0:
            raise ValueError(
                "The prompt's template exceeds the max context length of the model."
            )

        # Remove any in-context examples that would not fit
        in_context_examples_filtered: list[str] = []
        in_context_examples_token_counts_sum: int = 0
        for ice in in_context_examples:
            # Calculate how many tokens, *if* this in-context example is added
            # to the prompt
            ice_token_count = self.final_count_tokens(ice)
            if len(in_context_examples_filtered) > 0:
                next_sum = (
                    in_context_examples_token_counts_sum
                    + sep_token_count
                    + ice_token_count
                )
            else:
                next_sum = in_context_examples_token_counts_sum + ice_token_count

            # If there is room to add this in-context example to the prompt, add it
            if next_sum <= remaining_token_count:
                in_context_examples_filtered.append(ice)
                in_context_examples_token_counts_sum = next_sum
        in_context_examples = in_context_examples_filtered
        if provided_in_context_examples_length > 0 and len(in_context_examples) == 0:
            warnings.warn(
                f"Provided {provided_in_context_examples_length} in-context"
                " examples, but all were truncated due to length. This"
                " prompt will have no in-context examples.",
                stacklevel=2,
            )

        # Construct the final prompt
        final_prompt = sep.join(
            chain.from_iterable(
                [
                    ([beg_instruction] if beg_instruction is not None else []),
                    in_context_examples,
                    ([end_instruction] if end_instruction is not None else []),
                ]
            )
        )
        return final_prompt

    @abstractmethod
    def run(
        self,
        prompts: list[str],
        max_new_tokens: int = 999999999999999999,
        temperature: float = 1.0,
        top_p: float = 1.0,
        n: int = 1,
        stop: None | str | list[str] = None,
        repetition_penalty: None | float = None,
        best_of: int = 1,
        logit_bias: None | dict[int, float] = None,
        batch_size: int = 10,
    ) -> list[str] | list[list[str]]:
        args_cache_key = Hasher.hash(
            dict(
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                n=n,
                stop=stop,
                repetition_penalty=repetition_penalty,
                logit_bias=logit_bias,
            )
        )
        prompts_iter = iter(prompts)
        generated_texts = []
        while True:
            generated_texts_batch = []
            next(prompts_iter)
            prompts_batch = list(islice(prompts_iter, batch_size))
            if len(prompts_batch) == 0:
                break
            

            generated_texts.extend(
                self._run_batch(
                    args_cache_key=args_cache_key,
                    prompts=prompts_batch,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    n=n,
                    stop=stop,
                    repetition_penalty=repetition_penalty,
                    logit_bias=logit_bias,
                    batch_size=batch_size,
                )
            )
        return generated_texts

    @property
    def _cache_name(self) -> None | str:
        return None  # pragma: no cover


__all__ = ["LLM"]
