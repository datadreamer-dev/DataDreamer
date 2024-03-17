import datetime
import importlib
import json
import multiprocessing
import os
import sys
import uuid
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from math import ceil, floor
from random import Random
from time import sleep
from types import GeneratorType

import psutil
import pytest
import torch
from flaky import flaky
from sortedcontainers import SortedDict

from ... import DataDreamer
from ..._cachable._cachable import _is_primitive
from ...llms import (
    AI21,
    VLLM,
    Anthropic,
    Bedrock,
    Cohere,
    CTransformers,
    HFAPIEndpoint,
    HFTransformers,
    MistralAI,
    OpenAI,
    OpenAIAssistant,
    PaLM,
    ParallelLLM,
    Petals,
    Together,
    VertexAI,
)
from ...llms._chat_prompt_templates import (
    CHAT_PROMPT_TEMPLATES,
    SYSTEM_PROMPT_TYPES,
    SYSTEM_PROMPTS,
    _model_name_to_chat_prompt_template,
    _model_name_to_system_prompt,
)
from ...llms._litellm import LiteLLM
from ...llms.hf_transformers import CachedTokenizer
from ...llms.llm import _check_max_new_tokens_possible, _check_temperature_and_top_p
from ...utils.hf_model_utils import get_orig_model
from ...utils.import_utils import (
    ignore_litellm_warnings,
    ignore_transformers_warnings,
    import_module,
)
from ..test_utils.config import TEST_DIR

OPENAI_SKY_ANSWER = "The sky is typically blue during the day and black"
OPENAI_TREE_ANSWER = "Trees can be many different colors, depending on the"
OPENAI_SKY_CHAT_ANSWER = "The color of the sky can vary depending on the"
OPENAI_TREE_CHAT_ANSWER = "Trees can be various shades of green, depending on"


def _reload_pydantic():
    for m in list(sys.modules.keys()):
        if m.startswith("pydantic"):
            del sys.modules[m]


class TestLLM:
    def test_import(self):
        from concurrent import futures

        assert import_module("concurrent.futures") == futures
        with pytest.raises(ModuleNotFoundError):
            import_module("nonexistantmodule")

    def test_repr(self, create_datadreamer):
        assert (
            str(OpenAI("gpt-3.5-turbo-instruct")) == "<OpenAI (gpt-3.5-turbo-instruct)>"
        )

    def test_cache_created(self, create_datadreamer):
        assert OpenAI("gpt-3.5-turbo-instruct").cache_and_lock is None
        with create_datadreamer():
            db_path = os.path.join(
                DataDreamer.get_output_folder_path(),
                ".cache",
                "OpenAI_gpt-35-turbo-instruct_d943856c9b1e8f80.db",
            )
            db_lock_path = db_path + ".flock"
            llm = OpenAI("gpt-3.5-turbo-instruct")
            cache, cache_flock = llm.cache_and_lock  # type: ignore[misc]
            assert os.path.exists(db_path)
            with cache_flock:
                assert os.path.exists(db_lock_path)
            assert cache.filename == db_path
            assert cache.journal_mode == "WAL"
            assert not cache.autocommit
            assert cache_flock.lock_file == db_lock_path

    def test_cache_created_in_custom_location(self, create_datadreamer):
        custom_llm_cache_path_1 = os.path.join(
            TEST_DIR, ".cache" + "_" + uuid.uuid4().hex[0:10]
        )
        custom_llm_cache_path_2 = os.path.join(
            TEST_DIR, ".cache" + "_" + uuid.uuid4().hex[0:10]
        )

        db_path = os.path.join(
            custom_llm_cache_path_1, "OpenAI_gpt-35-turbo-instruct_d943856c9b1e8f80.db"
        )
        db_lock_path = db_path + ".flock"
        llm = OpenAI(
            "gpt-3.5-turbo-instruct", cache_folder_path=custom_llm_cache_path_1
        )
        cache, cache_flock = llm.cache_and_lock  # type: ignore[misc]
        assert os.path.exists(db_path)
        with cache_flock:
            assert os.path.exists(db_lock_path)
        assert cache.filename == db_path
        assert cache.journal_mode == "WAL"
        assert not cache.autocommit
        assert cache_flock.lock_file == db_lock_path

        with create_datadreamer():
            db_path = os.path.join(
                custom_llm_cache_path_2,
                "OpenAI_gpt-35-turbo-instruct_d943856c9b1e8f80.db",
            )
            db_lock_path = db_path + ".flock"
            llm = OpenAI(
                "gpt-3.5-turbo-instruct", cache_folder_path=custom_llm_cache_path_2
            )
            cache, cache_flock = llm.cache_and_lock  # type: ignore[misc]
            assert os.path.exists(db_path)
            with cache_flock:
                assert os.path.exists(db_lock_path)
            assert cache.filename == db_path
            assert cache.journal_mode == "WAL"
            assert not cache.autocommit
            assert cache_flock.lock_file == db_lock_path

    def test_llm_lock(self, create_datadreamer):
        with create_datadreamer():
            llm = OpenAI("gpt-3.5-turbo-instruct")
            with OpenAI("gpt-3.5-turbo-instruct"):
                pass

            with OpenAI("gpt-3.5-turbo-instruct"):
                with OpenAI("gpt-3.5-turbo-instruct"):
                    pass

            def acquire_lock(llm):
                with llm:
                    sleep(0.5)

            pool = ThreadPoolExecutor(max_workers=2)
            with pytest.raises(RuntimeError):
                list(pool.map(acquire_lock, [llm, llm]))

    def test_format_prompt(self, create_datadreamer):
        with create_datadreamer():
            llm = OpenAI("gpt-3.5-turbo-instruct")
            assert llm.get_max_context_length(max_new_tokens=0) == 4096
            assert (
                llm.format_prompt(
                    beg_instruction="Beg",
                    in_context_examples=["1", "2"],
                    end_instruction="End",
                )
                == "Beg\n1\n2\nEnd"
            )
            assert (
                llm.format_prompt(
                    beg_instruction=None, in_context_examples=None, end_instruction=None
                )
                == ""
            )
            assert (
                llm.format_prompt(
                    beg_instruction="Beg",
                    in_context_examples=None,
                    end_instruction="End",
                )
                == "Beg\nEnd"
            )
            assert (
                llm.format_prompt(
                    beg_instruction=None,
                    in_context_examples=["1", "2"],
                    end_instruction="End",
                )
                == "1\n2\nEnd"
            )
            assert (
                llm.format_prompt(
                    beg_instruction="Beg",
                    in_context_examples=["1", "2"],
                    end_instruction=None,
                )
                == "Beg\n1\n2"
            )
            assert (
                llm.format_prompt(
                    beg_instruction=None,
                    in_context_examples=["1", "2"],
                    end_instruction=None,
                )
                == "1\n2"
            )

    def test_format_prompt_error_instruction_too_large(self):
        llm = OpenAI("gpt-3.5-turbo-instruct")
        single_token = "aaaaaaaa"

        # Test just beginning instruction too large
        max_content_length = llm.get_max_context_length(max_new_tokens=0)
        llm.format_prompt(beg_instruction=single_token * (max_content_length - 1))
        llm.format_prompt(beg_instruction=single_token * (max_content_length))
        with pytest.raises(ValueError):
            llm.format_prompt(beg_instruction=single_token * (max_content_length + 1))

        # Test beg + end together too large
        # We do the "- 1" in max_content_length_left to account for the sep token
        max_content_length = llm.get_max_context_length(max_new_tokens=0)
        max_content_length_left = floor(max_content_length / 2.0) - 1
        max_content_length_right = ceil(max_content_length / 2.0)
        llm.format_prompt(
            beg_instruction=single_token * (max_content_length_left - 1),
            end_instruction=single_token * max_content_length_right,
        )
        llm.format_prompt(
            beg_instruction=single_token * (max_content_length_left),
            end_instruction=single_token * max_content_length_right,
        )
        with pytest.raises(ValueError):
            llm.format_prompt(
                beg_instruction=single_token * (max_content_length_left + 1),
                end_instruction=single_token * max_content_length_right,
            )

    def test_format_prompt_error_instruction_too_large_with_in_context_examples(self):
        llm = OpenAI("gpt-3.5-turbo-instruct")
        single_token = "aaaaaaaa"

        # Test just beginning instruction too large
        # We do the "- 1" in max_content_length to account for the sep token
        max_content_length = llm.get_max_context_length(max_new_tokens=0) - 1
        llm.format_prompt(
            beg_instruction=single_token * (max_content_length - 1),
            in_context_examples=[single_token],
        )
        with pytest.raises(ValueError):
            llm.format_prompt(
                beg_instruction=single_token * (max_content_length),
                in_context_examples=[single_token],
            )
        with pytest.raises(ValueError):
            llm.format_prompt(
                beg_instruction=single_token * (max_content_length + 1),
                in_context_examples=[single_token],
            )

        # Test beg + end together too large
        # We do the "- 2" in max_content_length_left to account for the sep tokens
        max_content_length = llm.get_max_context_length(max_new_tokens=0)
        max_content_length_left = floor(max_content_length / 2.0) - 2
        max_content_length_right = ceil(max_content_length / 2.0)
        llm.format_prompt(
            beg_instruction=single_token * (max_content_length_left - 1),
            in_context_examples=[single_token],
            end_instruction=single_token * max_content_length_right,
        )
        with pytest.raises(ValueError):
            llm.format_prompt(
                beg_instruction=single_token * (max_content_length_left),
                in_context_examples=[single_token],
                end_instruction=single_token * max_content_length_right,
            )
        with pytest.raises(ValueError):
            llm.format_prompt(
                beg_instruction=single_token * (max_content_length_left + 1),
                in_context_examples=[single_token],
                end_instruction=single_token * max_content_length_right,
            )

    def test_too_many_in_context_examples(self):
        llm = OpenAI("gpt-3.5-turbo-instruct")
        single_token = "aaaaaaaa"

        # Test too many in-context examples gets truncated
        # We do the "- 1" in max_content_length to account for the sep token
        max_content_length = llm.get_max_context_length(max_new_tokens=0) - 1
        assert (
            llm.format_prompt(
                beg_instruction=single_token * (max_content_length - 1),
                in_context_examples=["foo", "bar"],
            )
            == (single_token * (max_content_length - 1)) + "\n" + "foo"
        )

        # Test too many in-context examples gets truncated to those that can fit
        # We do the "- 1" in max_content_length to account for the sep token
        max_content_length = llm.get_max_context_length(max_new_tokens=0) - 1
        assert (
            llm.format_prompt(
                beg_instruction=single_token * (max_content_length - 1),
                in_context_examples=["This is a longer example that won't fit", "bar"],
            )
            == (single_token * (max_content_length - 1)) + "\n" + "bar"
        )

        # Test too many in-context examples gets truncated to none
        # We do the "- 1" in max_content_length to account for the sep token
        max_content_length = llm.get_max_context_length(max_new_tokens=0) - 1
        with pytest.raises(ValueError):
            assert llm.format_prompt(
                beg_instruction=single_token * (max_content_length - 1),
                in_context_examples=["This is a longer example that won't fit"],
            ) == (single_token * (max_content_length - 1))

    def test_is_primitive(self):
        assert _is_primitive("foo")
        assert _is_primitive(0.5)
        assert _is_primitive(True)
        assert _is_primitive(5)
        assert not _is_primitive(object())
        assert _is_primitive((1 == 5))
        assert _is_primitive([1 == 5])
        assert _is_primitive((1 == object()))
        assert _is_primitive({"foo": 5})
        assert not _is_primitive({"foo": object()})

    def test_check_temperature_and_top_p(self):
        assert _check_temperature_and_top_p(
            temperature=0.3,
            top_p=0.4,
            supports_zero_temperature=True,
            supports_zero_top_p=True,
        ) == (0.3, 0.4)
        assert _check_temperature_and_top_p(
            temperature=0.0,
            top_p=0.0,
            supports_zero_temperature=True,
            supports_zero_top_p=True,
        ) == (0.0, 0.0)
        assert _check_temperature_and_top_p(
            temperature=0.0,
            top_p=1.0,
            supports_zero_temperature=False,
            supports_zero_top_p=True,
        ) == (1.0, 0.0)
        assert _check_temperature_and_top_p(
            temperature=0.7,
            top_p=0.0,
            supports_zero_temperature=True,
            supports_zero_top_p=False,
        ) == (1.0, 0.001)
        assert _check_temperature_and_top_p(
            temperature=0.3,
            top_p=0.4,
            supports_zero_temperature=False,
            supports_zero_top_p=False,
        ) == (0.3, 0.4)
        assert _check_temperature_and_top_p(
            temperature=0.0,
            top_p=1.0,
            supports_zero_temperature=False,
            supports_zero_top_p=False,
        ) == (1.0, 0.001)
        assert _check_temperature_and_top_p(
            temperature=0.3,
            top_p=1.0,
            supports_zero_temperature=False,
            supports_zero_top_p=False,
            supports_one_top_p=False,
        ) == (0.3, 0.999)

    def test_check_max_new_tokens_possible(create_datadreamer):
        # Check max output length
        llm = OpenAI("gpt-4")
        assert _check_max_new_tokens_possible(
            self=llm,
            max_length_func=lambda prompts: 100,
            prompts=[],
            max_new_tokens=None,
        ) == (8174 - 100)
        assert (
            _check_max_new_tokens_possible(
                self=llm,
                max_length_func=lambda prompts: 100,
                prompts=[],
                max_new_tokens=5000,
            )
            == 5000
        )
        llm = OpenAI("gpt-4-turbo-preview")
        assert (
            _check_max_new_tokens_possible(
                self=llm,
                max_length_func=lambda prompts: 100,
                prompts=[],
                max_new_tokens=None,
            )
            == 4096
        )
        assert (
            _check_max_new_tokens_possible(
                self=llm,
                max_length_func=lambda prompts: 100,
                prompts=[],
                max_new_tokens=4096,
            )
            == 4096
        )
        # Make sure an error is thrown if the model's output length is surpassed
        with pytest.raises(ValueError):
            assert (
                _check_max_new_tokens_possible(
                    self=llm,
                    max_length_func=lambda prompts: 100,
                    prompts=[],
                    max_new_tokens=5000,
                )
                == 4096
            )

    def test_run_with_no_prompts(self, create_datadreamer):
        llm = OpenAI("gpt-3.5-turbo-instruct")
        generated_texts = llm.run(
            [],
            max_new_tokens=25,
            temperature=0.0,
            top_p=1.0,
            n=1,
            repetition_penalty=None,
            logit_bias=None,
            batch_size=2,
        )
        assert generated_texts == []

    def test_run_with_invalid_batch_size(self, create_datadreamer):
        llm = OpenAI("gpt-3.5-turbo-instruct")
        with pytest.raises(AssertionError):
            llm.run(
                ["What color is the sky?"],
                max_new_tokens=25,
                temperature=0.0,
                top_p=1.0,
                n=1,
                repetition_penalty=None,
                logit_bias=None,
                batch_size=-1,
            )

    def test_run_with_repeated_prompts_with_cache(self, create_datadreamer, mocker):
        prompt_1 = "What color is the sky?"
        prompt_2 = "What color is grass?"
        prompt_3 = "What color are apples?"
        prompt_4 = "What color are pears?"
        prompts = [
            prompt_1,
            prompt_1,
            prompt_1,
            prompt_1,
            prompt_2,
            prompt_3,
            prompt_3,
            prompt_1,
        ]

        def _run_batch_mocked(**kwargs):
            return [f"Response to: {p}" for p in kwargs["inputs"]]

        with create_datadreamer():
            llm = OpenAI("gpt-3.5-turbo-instruct")
            mocker.patch.object(llm, "_run_batch", side_effect=_run_batch_mocked)

            # Test that with repeated prompts with cache, only new prompts should get
            # sent to the LLM
            generated_texts = llm.run(
                prompts,
                max_new_tokens=25,
                temperature=0.0,
                top_p=1.0,
                n=1,
                repetition_penalty=None,
                logit_bias=None,
                batch_size=2,
                batch_scheduler_buffer_size=2,
                adaptive_batch_size=True,
            )
            assert generated_texts == [f"Response to: {p}" for p in prompts]
            cache_query = "SELECT key, value FROM run_cache WHERE key IN (?, ?, ?, ?)"
            cache, _ = llm.cache_and_lock  # type: ignore[misc]
            assert sorted(
                list(
                    cache.conn.select(
                        cache_query,
                        [
                            "8dadfe9d20ba6f71",
                            "3115a45697fbc214",
                            "e324e48c7e9a3302",
                            "adaptive_batch_sizes",
                        ],
                    )
                )
            ) == [
                ("3115a45697fbc214", cache.encode(f"Response to: {prompt_2}")),
                ("8dadfe9d20ba6f71", cache.encode(f"Response to: {prompt_1}")),
                (
                    "adaptive_batch_sizes",
                    cache.encode(
                        defaultdict(
                            SortedDict,
                            {
                                ((psutil.virtual_memory().total, ()), 25): SortedDict(
                                    {678: Counter({2: 2})}
                                )
                            },
                        )
                    ),
                ),
                ("e324e48c7e9a3302", cache.encode(f"Response to: {prompt_3}")),
            ]
            assert llm._run_batch.call_count == 2  # type: ignore[attr-defined]
            assert llm._run_batch.call_args_list[0].kwargs["inputs"] == [  # type: ignore[attr-defined]
                prompt_1,
                prompt_1,
            ]
            assert llm._run_batch.call_args_list[1].kwargs["inputs"] == [  # type: ignore[attr-defined]
                prompt_2,
                prompt_3,
            ]

            # Make sure the cache is used even after reloading the LLM
            llm = OpenAI("gpt-3.5-turbo-instruct")
            mocker.patch.object(llm, "_run_batch", side_effect=_run_batch_mocked)
            generated_texts = llm.run(
                prompts + [prompt_4],
                max_new_tokens=25,
                temperature=0.0,
                top_p=1.0,
                n=1,
                repetition_penalty=None,
                logit_bias=None,
                batch_size=2,
                batch_scheduler_buffer_size=2,
                adaptive_batch_size=True,
            )
            assert llm._run_batch.call_count == 1  # type: ignore[attr-defined]
            assert llm._run_batch.call_args_list[0].kwargs["inputs"] == [  # type: ignore[attr-defined]
                prompt_4
            ]

            # Make sure the cache is not used if force=True
            llm = OpenAI("gpt-3.5-turbo-instruct")
            mocker.patch.object(llm, "_run_batch", side_effect=_run_batch_mocked)
            generated_texts = llm.run(
                prompts,
                max_new_tokens=25,
                temperature=0.0,
                top_p=1.0,
                n=1,
                repetition_penalty=None,
                logit_bias=None,
                batch_size=2,
                batch_scheduler_buffer_size=2,
                adaptive_batch_size=True,
                force=True,
            )
            assert llm._run_batch.call_count == 4  # type: ignore[attr-defined]

            # Make sure RuntimeError is thrown if a never seen prompt is encountered.
            with pytest.raises(RuntimeError):
                llm.run(
                    ["Never seen prompt."],
                    max_new_tokens=25,
                    temperature=0.0,
                    top_p=1.0,
                    n=1,
                    repetition_penalty=None,
                    logit_bias=None,
                    batch_size=2,
                    batch_scheduler_buffer_size=2,
                    adaptive_batch_size=True,
                    cache_only=True,
                )

    def test_run_with_repeated_prompts_without_cache(self, create_datadreamer, mocker):
        prompt = "What color is the sky?"
        prompts = [prompt] * 4

        def _run_batch_mocked(**kwargs):
            return [f"Response to: {p}" for p in kwargs["inputs"]]

        llm = OpenAI("gpt-3.5-turbo-instruct")
        mocker.patch.object(llm, "_run_batch", side_effect=_run_batch_mocked)

        # Test repeated prompts without cache, get sent to the LLM
        generated_texts = llm.run(
            prompts,
            max_new_tokens=25,
            temperature=0.0,
            top_p=1.0,
            n=1,
            repetition_penalty=None,
            logit_bias=None,
            batch_size=2,
        )
        assert generated_texts == [f"Response to: {p}" for p in prompts]
        assert llm._run_batch.call_count == 2  # type: ignore[attr-defined]
        assert llm._run_batch.call_args_list[0].kwargs["inputs"] == [prompt] * 2  # type: ignore[attr-defined]
        assert llm._run_batch.call_args_list[1].kwargs["inputs"] == [prompt] * 2  # type: ignore[attr-defined]

    def test_error_in_run_propagates(self, create_datadreamer, mocker):
        def _run_batch_mocked(*args, **kwargs):
            raise RuntimeError("RandomError")

        with create_datadreamer():
            llm = HFTransformers("google/flan-t5-small")
            mocker.patch.object(llm, "_run_batch", side_effect=_run_batch_mocked)

            with pytest.raises(RuntimeError):
                llm.run(
                    ["test"],
                    max_new_tokens=25,
                    temperature=0.0,
                    top_p=0.0,
                    n=1,
                    batch_size=1,
                )
            assert llm._run_batch.call_count == 1  # type: ignore[attr-defined]

    def test_adaptive_batch_size_and_batch_scheduler(self, create_datadreamer, mocker):
        with create_datadreamer():
            llm = HFTransformers("google/flan-t5-small")

            def _run_batch_mocked(*args, **kwargs):
                cached_tokenizer = kwargs["cached_tokenizer"]
                num_tokens = max(
                    [
                        len(
                            cached_tokenizer(
                                p,
                                **{
                                    "padding": False,
                                    "add_special_tokens": False,
                                    "return_tensors": "pt",
                                },
                            )["input_ids"][0]
                        )
                        for p in kwargs["inputs"]
                    ]
                )
                batch_size = len(kwargs["inputs"])
                max_batch_size_before_oom = 110.209 - 0.208984 * num_tokens
                if batch_size > max_batch_size_before_oom:
                    raise torch.cuda.OutOfMemoryError("Ran out of GPU Memory.")
                else:
                    return ["Response " + p.split(" ")[1] for p in kwargs["inputs"]]

            mocker.patch.object(llm, "_run_batch", side_effect=_run_batch_mocked)

            # Test adaptive batch size retry logic
            r = Random(42)
            num_words = [r.choice(range(100)) for _ in range(10000)]
            generated_texts = llm.run(
                [
                    f"{i} " + " ".join([f"test_{num_words}"] * num_words)
                    for i, num_words in enumerate(num_words)
                ],
                max_new_tokens=25,
                temperature=0.0,
                top_p=0.0,
                n=1,
                batch_size=200,
            )
            assert isinstance(generated_texts, list)
            assert len(generated_texts) == len(num_words)
            assert generated_texts[0] == f"Response test_{num_words[0]}"
            assert generated_texts[-1] == f"Response test_{num_words[-1]}"
            key = list(llm.adaptive_batch_sizes.keys())[0]
            assert {k: dict(v) for k, v in llm.adaptive_batch_sizes[key].items()} == {
                85: {105: 3, 94: 14, 71: 5},
                170: {84: 3, 75: 3, 71: 14},
                255: {67: 3, 60: 14, 62: 2},
                340: {54: 4, 48: 15, 43: 5},
            }
            llm.reset_adaptive_batch_sizing()
            assert len(llm.adaptive_batch_sizes[key]) == 0

    def test_cached_tokenizer_during_adaptive_batch_retry(
        self, create_datadreamer, mocker
    ):
        with create_datadreamer():
            llm = HFTransformers("google/flan-t5-small")
            old_run_batch = llm._run_batch
            cached_tokenizer = None

            def _run_batch_mocked(*args, **kwargs):
                nonlocal cached_tokenizer
                cached_tokenizer = kwargs["cached_tokenizer"]
                if llm._run_batch.call_count == 1:  # type: ignore[attr-defined]
                    raise torch.cuda.OutOfMemoryError("Ran out of GPU Memory.")
                else:
                    return old_run_batch(*args, **kwargs)

            mocker.patch.object(llm, "_run_batch", side_effect=_run_batch_mocked)

            # Check cached tokenizer works with encoder-decoder model
            llm.run(
                ["prompt1", "prompt2", "prompt3", "prompt4"],
                max_new_tokens=25,
                temperature=0.0,
                top_p=0.0,
                n=1,
                batch_size=6,
                batch_scheduler_buffer_size=20,
            )
            assert isinstance(cached_tokenizer, CachedTokenizer)
            assert set(cached_tokenizer.cache.keys()) == set(
                [
                    "a8bd7240552809a7",
                    "e9d81dcd6a3ab79e",
                    "2d7fdfe259554a4f",
                    "9b7d1605c56f136b",
                    "a58112238357b7ba",
                ]
            )

            llm = HFTransformers("gpt2")
            old_run_batch = llm._run_batch
            cached_tokenizer = None

            def _run_batch_mocked(*args, **kwargs):
                nonlocal cached_tokenizer
                cached_tokenizer = kwargs["cached_tokenizer"]
                if llm._run_batch.call_count == 1:  # type: ignore[attr-defined]
                    raise torch.cuda.OutOfMemoryError("Ran out of GPU Memory.")
                else:
                    return old_run_batch(*args, **kwargs)

            mocker.patch.object(llm, "_run_batch", side_effect=_run_batch_mocked)

            # Check cached tokenizer works with decoder-only model
            llm.run(
                ["prompt1", "prompt2", "prompt3", "prompt4"],
                max_new_tokens=25,
                temperature=0.0,
                top_p=0.0,
                n=1,
                batch_size=6,
                batch_scheduler_buffer_size=20,
            )
            assert isinstance(cached_tokenizer, CachedTokenizer)
            assert set(cached_tokenizer.cache.keys()) == set(
                [
                    "a8bd7240552809a7",
                    "e9d81dcd6a3ab79e",
                    "2d7fdfe259554a4f",
                    "9b7d1605c56f136b",
                ]
            )

    def test_batch_size_causes_oom_error(self, create_datadreamer, mocker):
        def _run_batch_mocked(*args, **kwargs):
            raise torch.cuda.OutOfMemoryError("Ran out of GPU Memory.")

        with create_datadreamer():
            llm = HFTransformers("google/flan-t5-small")
            mocker.patch.object(llm, "_run_batch", side_effect=_run_batch_mocked)

            # If OOM error is constantly thrown, we will keep trying smaller batch sizes
            # until we get to a batch size of 1, at which point the OOM error is raised
            # to the user
            with pytest.raises(torch.cuda.OutOfMemoryError):
                llm.run(
                    [
                        f"{i} " + " ".join([f"test_{num_words}"] * num_words)
                        for i, num_words in enumerate(([19] * 6))
                    ],
                    max_new_tokens=25,
                    temperature=0.0,
                    top_p=0.0,
                    n=1,
                    batch_size=6,
                    batch_scheduler_buffer_size=20,
                )
            assert llm._run_batch.call_count == 6  # type: ignore[attr-defined]

    def test_disable_batch_scheduler(self, create_datadreamer, mocker):
        prompts = [
            "Question: What color is the sky?\nAnswer:",
            "Question: What color are trees?\nAnswer:",
        ]

        def _run_batch_mocked(**kwargs):
            return [f"Response to: {p}" for p in kwargs["inputs"]]

        with create_datadreamer():
            llm = HFTransformers("google/flan-t5-small")
            mocker.patch.object(llm, "_run_batch", side_effect=_run_batch_mocked)

            # If batch scheduling is enabled, prompts are run from shortest in length
            # to longest in length (reverse order)
            generated_texts = llm.run(
                prompts,
                max_new_tokens=25,
                temperature=0.0,
                top_p=0.0,
                n=1,
                stop="Question:",
                repetition_penalty=None,
                logit_bias=None,
                batch_size=2,
                batch_scheduler_buffer_size=2,
                force=True,
            )
            assert generated_texts == [f"Response to: {p}" for p in prompts]
            assert llm._run_batch.call_args_list[0].kwargs["inputs"] == (  # type: ignore[attr-defined]
                list(reversed(prompts))
            )

            # If batch scheduling is disabled, prompts are run in the original order
            generated_texts = llm.run(
                prompts,
                max_new_tokens=25,
                temperature=0.0,
                top_p=0.0,
                n=1,
                stop="Question:",
                repetition_penalty=None,
                logit_bias=None,
                batch_size=2,
                batch_scheduler_buffer_size=-1,
                force=True,
            )
            assert generated_texts == [f"Response to: {p}" for p in prompts]
            assert llm._run_batch.call_args_list[1].kwargs["inputs"] == (  # type: ignore[attr-defined]
                prompts
            )

    def test_disable_adaptive_batch_size(self, create_datadreamer, mocker):
        with create_datadreamer():
            llm = HFTransformers("gpt2")
            old_run_batch = llm._run_batch

            def _run_batch_mocked(*args, **kwargs):
                if llm._run_batch.call_count == 1:  # type: ignore[attr-defined]
                    raise torch.cuda.OutOfMemoryError("Ran out of GPU Memory.")
                else:
                    return old_run_batch(*args, **kwargs)

            # If adaptive_batch_size=False, we cannot recover from OOM error
            mocker.patch.object(llm, "_run_batch", side_effect=_run_batch_mocked)
            with pytest.raises(torch.cuda.OutOfMemoryError):
                llm.run(
                    ["prompt1", "prompt2", "prompt3", "prompt4"],
                    max_new_tokens=25,
                    temperature=0.0,
                    top_p=0.0,
                    n=1,
                    batch_size=6,
                    batch_scheduler_buffer_size=20,
                    adaptive_batch_size=False,
                )

            # If adaptive_batch_size=False, llm.adaptive_batch_sizes should be empty
            # after running
            llm = HFTransformers("gpt2")
            generated_texts = llm.run(
                ["prompt1", "prompt2", "prompt3", "prompt4"],
                max_new_tokens=25,
                temperature=0.0,
                top_p=0.0,
                n=1,
                batch_size=6,
                batch_scheduler_buffer_size=20,
                adaptive_batch_size=False,
            )
            assert isinstance(generated_texts, list)
            assert len(generated_texts) == 4
            assert llm.adaptive_batch_sizes == defaultdict(SortedDict, {})

    def test_progress_logging(self, create_datadreamer, mocker, caplog):
        def _run_batch_mocked(**kwargs):
            return [f"Response to: {p}" for p in kwargs["inputs"]]

        with create_datadreamer():
            llm = HFTransformers("gpt2")
            mocker.patch.object(llm, "_run_batch", side_effect=_run_batch_mocked)

            # We should get progress logs
            llm.run(
                (str(i) for i in range(10)),
                max_new_tokens=25,
                temperature=0.0,
                top_p=1.0,
                n=1,
                repetition_penalty=None,
                logit_bias=None,
                batch_size=2,
                progress_interval=0,
            )
            logs = [rec.message for rec in caplog.records]
            caplog.clear()
            assert len(logs) == 7
            assert logs[3] == "Progress: 4 prompt(s) 🔄"

            # With total_num_prompts specified, we should have logs with a %
            llm.run(
                (str(i) for i in range(10)),
                max_new_tokens=25,
                temperature=0.0,
                top_p=1.0,
                n=1,
                repetition_penalty=None,
                logit_bias=None,
                batch_size=2,
                progress_interval=0,
                total_num_prompts=10,
            )
            logs = [rec.message for rec in caplog.records]
            caplog.clear()
            assert len(logs) == 2
            assert "Progress: 100% 🔄" in logs[1]

            # With disable verbosity, we should have zero logs
            llm.run(
                (str(i) for i in range(10)),
                max_new_tokens=25,
                temperature=0.0,
                top_p=1.0,
                n=1,
                repetition_penalty=None,
                logit_bias=None,
                batch_size=2,
                progress_interval=0,
                total_num_prompts=10,
                verbose=False,
            )

            logs = [rec.message for rec in caplog.records]
            caplog.clear()
            assert len(logs) == 0

    def test_chat_prompt_templates(self):
        # Validate format
        for system_prompt_type in SYSTEM_PROMPT_TYPES.values():
            assert system_prompt_type in SYSTEM_PROMPTS
        for (
            chat_prompt_template_type,
            chat_prompt_template,
        ) in CHAT_PROMPT_TEMPLATES.items():
            assert chat_prompt_template.count("{") == chat_prompt_template.count("}")
            assert "{{prompt}}" in chat_prompt_template
            if "{{system_prompt}}" in chat_prompt_template:
                assert chat_prompt_template_type in SYSTEM_PROMPT_TYPES

        # Test a few model names
        assert (
            _model_name_to_chat_prompt_template("datadreamer/test")
            == "[INST] <<SYS>>\n{{system_prompt}}\n<</SYS>>\n\n{{prompt}} [/INST] "
        )
        assert (
            _model_name_to_system_prompt(
                chat_prompt_template="{{system_prompt}} {{prompt}}",
                model_name="datadreamer/test",
            )
            == "You are a helpful assistant."
        )
        assert (
            _model_name_to_chat_prompt_template("datadreamer/test-guanaco")
            == "### Human: {{prompt}}\n### Assistant: "
        )
        assert (
            _model_name_to_system_prompt(
                chat_prompt_template="{{system_prompt}} {{prompt}}",
                model_name="datadreamer/test-guanaco",
            )
            is None
        )

    # skipping because of https://github.com/huggingface/transformers/pull/26765"
    # def test_hf_chat_prompt_template_and_system_prompt(self, create_datadreamer):
    #     with create_datadreamer():
    #         assert _chat_prompt_template_and_system_prompt("asdasdas") is None
    #         assert _chat_prompt_template_and_system_prompt("gpt2") is None
    #         assert _chat_prompt_template_and_system_prompt("t5-small") is None
    #         assert (
    #             _chat_prompt_template_and_system_prompt("t5-small", revision="asdasdas")
    #             is None
    #         )
    #         assert _chat_prompt_template_and_system_prompt(
    #             "meta-llama/Llama-2-7b-chat-hf"
    #         ) == (CHAT_PROMPT_TEMPLATES["llama_system"], SYSTEM_PROMPTS["llama_system"])
    #         assert _chat_prompt_template_and_system_prompt(
    #             "mistralai/Mistral-7B-Instruct-v0.1"
    #         ) == (CHAT_PROMPT_TEMPLATES["llama"], None)


class TestOpenAI:
    def test_init(self, create_datadreamer):
        with create_datadreamer():
            db_path = os.path.join(
                DataDreamer.get_output_folder_path(),
                ".cache",
                "OpenAI_gpt-35-turbo-instruct_d943856c9b1e8f80.db",
            )
            llm = OpenAI("gpt-3.5-turbo-instruct")
            cache, _ = llm.cache_and_lock  # type: ignore[misc]
            assert os.path.exists(db_path)

    def test_metadata(self, create_datadreamer):
        llm = OpenAI("gpt-3.5-turbo-16k")
        assert llm.model_card == (
            "https://github.com/openai/gpt-3/blob/"
            "d7a9bb505df6f630f9bab3b30c889e52f22eb9ea/model-card.md"
        )
        assert llm.license == "https://openai.com/policies"
        assert isinstance(llm.citation, list)
        assert len(llm.citation) == 2
        assert llm.citation[0].startswith("@article{brown2020language")
        assert llm.citation[0].endswith("year={2020}\n}")
        assert llm.citation[1].startswith("@article{ouyang2022training")
        assert llm.citation[1].endswith("year={2022}\n}")
        llm = OpenAI("gpt-4")
        assert llm.model_card == "https://cdn.openai.com/papers/gpt-4-system-card.pdf"
        assert llm.license == "https://openai.com/policies"
        assert isinstance(llm.citation, list)
        assert len(llm.citation) == 2
        assert llm.citation[0].startswith("@article{OpenAI2023GPT4TR,")
        assert llm.citation[0].endswith(
            "url={https://api.semanticscholar.org/CorpusID:257532815}\n}"
        )
        assert llm.citation[1].startswith("@article{ouyang2022training")
        assert llm.citation[1].endswith("year={2022}\n}")

    def test_count_tokens(self, create_datadreamer):
        with create_datadreamer():
            llm = OpenAI("gpt-3.5-turbo-instruct")
            assert llm.count_tokens("This is a test.") == 5

    def test_get_max_context_length(self, create_datadreamer):
        with create_datadreamer():
            # Check max context length
            llm = OpenAI("gpt-4")
            assert llm.get_max_context_length(max_new_tokens=0) == 8174
            llm = OpenAI("gpt-4-turbo-preview")
            assert llm.get_max_context_length(max_new_tokens=0) == 127982
            llm = OpenAI("gpt-3.5-turbo")
            assert llm.get_max_context_length(max_new_tokens=0) == 16367
            llm = OpenAI("gpt-3.5-turbo-instruct")
            assert llm.get_max_context_length(max_new_tokens=0) == 4096

    def test_get_max_output_length(self, create_datadreamer):
        with create_datadreamer():
            # Check max output length
            llm = OpenAI("gpt-4")
            assert llm._get_max_output_length() is None
            llm = OpenAI("gpt-4-turbo-preview")
            assert llm._get_max_output_length() == 4096
            llm = OpenAI("gpt-3.5-turbo")
            assert llm._get_max_output_length() == 4096
            llm = OpenAI("gpt-3.5-turbo-instruct")
            assert llm._get_max_output_length() is None

    @pytest.mark.skipif(
        "OPENAI_API_KEY" not in os.environ, reason="requires OpenAI API key"
    )
    def test_length_errors(self, create_datadreamer, mocker):
        with create_datadreamer():
            llm = OpenAI("gpt-3.5-turbo-instruct")

            # Make sure if max_new_tokens=None, it gets automatically calculated
            def create_mocked(*args, **kwargs):
                raise RuntimeError("Skip running.")

            mocker.patch.object(
                llm.client.completions, "create", side_effect=create_mocked
            )
            with pytest.raises(RuntimeError):
                llm.run(
                    ["A short prompt."],
                    max_new_tokens=None,
                    temperature=0.0,
                    top_p=0.0,
                    n=1,
                    repetition_penalty=None,
                    logit_bias=None,
                    batch_size=1,
                )
            assert (
                llm.client.completions.create.call_args_list[0].kwargs["max_tokens"]  # type: ignore[attr-defined]
                == 4092
            )

            # Make sure an error is thrown if the model's context length is surpassed
            with pytest.raises(ValueError):
                llm.run(
                    ["A too long prompt. " * 4096],
                    max_new_tokens=25,
                    temperature=0.0,
                    top_p=0.0,
                    n=1,
                    repetition_penalty=None,
                    logit_bias=None,
                    batch_size=1,
                )

    @pytest.mark.skipif(
        "OPENAI_API_KEY" not in os.environ, reason="requires OpenAI API key"
    )
    def test_run(self, create_datadreamer):
        with create_datadreamer():
            llm = OpenAI("gpt-3.5-turbo-instruct")

            # Test Completion model
            generated_texts = llm.run(
                [
                    "Question: What color is the sky?\nAnswer:",
                    "Question: What color are trees?\nAnswer:",
                ],
                max_new_tokens=10,
                temperature=0.0,
                top_p=1.0,
                n=2,
                stop=["Question:"],
                repetition_penalty=None,
                logit_bias=None,
                batch_size=2,
            )
            assert generated_texts == [
                [OPENAI_SKY_ANSWER] * 2,
                [OPENAI_TREE_ANSWER] * 2,
            ]

            # Test return_generator
            generated_texts = llm.run(
                [
                    "Question: What color is the sky?\nAnswer:",
                    "Question: What color are trees?\nAnswer:",
                ],
                max_new_tokens=10,
                temperature=0.0,
                top_p=1.0,
                n=2,
                stop=["Question:"],
                repetition_penalty=None,
                logit_bias=None,
                batch_size=2,
                return_generator=True,
            )
            assert isinstance(generated_texts, GeneratorType)
            assert list(generated_texts) == [
                [OPENAI_SKY_ANSWER] * 2,
                [OPENAI_TREE_ANSWER] * 2,
            ]

            # Test ChatCompletion model (gpt-3.5-turbo) and test n=1
            llm = OpenAI("gpt-3.5-turbo")
            generated_texts = llm.run(
                ["What color is the sky?", "What color are trees?"],
                max_new_tokens=10,
                temperature=0.0,
                top_p=1.0,
                n=1,
                stop=". ",
                repetition_penalty=None,
                logit_bias=None,
                batch_size=2,
            )
            assert generated_texts == [OPENAI_SKY_CHAT_ANSWER, OPENAI_TREE_CHAT_ANSWER]

            # Test unload model
            assert "client" in llm.__dict__ and "tokenizer" in llm.__dict__
            llm.unload_model()
            assert "client" not in llm.__dict__ and "tokenizer" not in llm.__dict__


class TestOpenAIAssistant:
    def test_init(self, create_datadreamer):
        with create_datadreamer():
            db_path = os.path.join(
                DataDreamer.get_output_folder_path(),
                ".cache",
                "OpenAIAssistant_gpt-35-turbo-1106_7bb0af8dbb968cf6.db",
            )
            llm = OpenAIAssistant(
                "gpt-3.5-turbo-1106",
                tools=[{"type": "retrieval"}, {"type": "code_interpreter"}],
            )
            cache, _ = llm.cache_and_lock  # type: ignore[misc]
            assert os.path.exists(db_path)

    @pytest.mark.skipif(
        "OPENAI_API_KEY" not in os.environ, reason="requires OpenAI API key"
    )
    @flaky(max_runs=3)
    def test_run(self, create_datadreamer):
        with create_datadreamer():
            llm = OpenAIAssistant("gpt-4", tools=[{"type": "code_interpreter"}])
            generated_texts = llm.run(
                ["What is 128532 + 850283?", "What is today's date?"], batch_size=2
            )
            assert isinstance(generated_texts, list)
            assert len(generated_texts) == 2
            assert "978815" in generated_texts[0]
            date = datetime.datetime.now()
            month_str = date.strftime("%b")
            day = int(date.strftime("%d").lstrip("0"))
            # To account for time-zone difference between server & OpenAI's servers
            # we do +1 / -1
            day_strs = [str(d) for d in [day - 1, day, day + 1]]
            year_str = date.strftime("%Y")
            assert (
                month_str in generated_texts[1]
                and any(day_str in generated_texts[1] for day_str in day_strs)
                and year_str in generated_texts[1]
            )

            # Test unload model
            assert (
                "client" in llm.__dict__
                and "tokenizer" in llm.__dict__
                and "assistant_id" in llm.__dict__
            )
            llm.unload_model()
            assert (
                "client" not in llm.__dict__
                and "tokenizer" not in llm.__dict__
                and "assistant_id" not in llm.__dict__
            )


class TestHFTransformers:
    def test_init(self, create_datadreamer):
        with create_datadreamer():
            with pytest.raises(ValueError):
                llm = HFTransformers(
                    "google/flan-t5-small", system_prompt="You are a helpful assistant."
                )
            with pytest.raises(ValueError):
                llm = HFTransformers(
                    "google/flan-t5-small",
                    chat_prompt_template="{{system_prompt}} {{prompt}}",
                    system_prompt=None,
                )
            with pytest.raises(ValueError):
                llm = HFTransformers(
                    "google/flan-t5-small", adapter_kwargs={"revision": "other-branch"}
                )
            llm = HFTransformers(
                "google/flan-t5-small",
                revision="2d036ee774a9cb8d7e03c9f2e78ae0a16343a9d9",
                device_map="auto",
            )
            db_path = os.path.join(
                DataDreamer.get_output_folder_path(),
                ".cache",
                "HFTransformers_google-flan-t5-small_2d036ee774a9cb8d7e03c9f2e78ae0a16343a9d9_torch.float32.db",
            )
            cache, _ = llm.cache_and_lock  # type: ignore[misc]
            assert os.path.exists(db_path)
            llm = HFTransformers(
                "google/flan-t5-small",
                revision="2d036ee774a9cb8d7e03c9f2e78ae0a16343a9d9",
                device_map="auto",
                dtype="torch.float16",
            )
            db_path = os.path.join(
                DataDreamer.get_output_folder_path(),
                ".cache",
                "HFTransformers_google-flan-t5-small_2d036ee774a9cb8d7e03c9f2e78ae0a16343a9d9_torch.float16.db",
            )
            cache, _ = llm.cache_and_lock  # type: ignore[misc]
            assert os.path.exists(db_path)

    def test_metadata(self, create_datadreamer):
        with create_datadreamer():
            llm = HFTransformers(
                "google/flan-t5-small",
                revision="2d036ee774a9cb8d7e03c9f2e78ae0a16343a9d9",
                device_map="auto",
            )
            assert llm.model_card == "https://huggingface.co/google/flan-t5-small"
            assert llm.license == "apache-2.0"
            assert isinstance(llm.citation, list)
            assert len(llm.citation) == 2
            assert llm.citation[-1].startswith(
                "@misc{https://doi.org/10.48550/arxiv.2210.11416"
            )
            assert llm.citation[-1].endswith(
                "{Creative Commons Attribution 4.0 International}\n}"
            )
            llm = HFTransformers(
                "stabilityai/StableBeluga2",
                revision="effbf4ce2957180eed1497a29c7fdcc2129f2671",
                device_map="auto",
            )
            assert llm.license == (
                "https://huggingface.co/stabilityai/StableBeluga2/blob/"
                "effbf4ce2957180eed1497a29c7fdcc2129f2671/LICENSE.txt"
            )
            assert isinstance(llm.citation, list)
            assert len(llm.citation) == 4
            llm = HFTransformers("lvwerra/gpt2-imdb")
            assert isinstance(llm.citation, list)
            assert len(llm.citation) == 1

    def test_count_tokens(self, create_datadreamer):
        with create_datadreamer():
            llm = HFTransformers("google/flan-t5-small")
            assert llm.count_tokens("This is a test.") == 7

    def test_get_max_context_length(self, create_datadreamer):
        with create_datadreamer():
            llm = HFTransformers("google/flan-t5-small")
            assert llm.get_max_context_length(max_new_tokens=0) == 512
        with create_datadreamer():
            llm = HFTransformers("facebook/opt-125m")
            assert llm.get_max_context_length(max_new_tokens=0) == 2048
        with create_datadreamer():
            llm = HFTransformers("NousResearch/Yarn-Mistral-7b-128k")
            assert llm.get_max_context_length(max_new_tokens=0) == 131072

    def test_length_errors(self, create_datadreamer, mocker):
        with create_datadreamer():
            llm = HFTransformers("gpt2")

            # Make sure if max_new_tokens=None, it gets automatically calculated
            def generate_mocked(*args, **kwargs):
                raise RuntimeError("Skip running.")

            mocker.patch.object(llm.model, "generate", side_effect=generate_mocked)
            with pytest.raises(RuntimeError):
                llm.run(
                    ["A short prompt."],
                    max_new_tokens=None,
                    temperature=0.0,
                    top_p=0.0,
                    n=1,
                    repetition_penalty=None,
                    logit_bias=None,
                    batch_size=1,
                )
            assert llm.model.generate.call_args_list[0].kwargs["max_new_tokens"] == 1020

            # Make sure an error is thrown if the model's context length is surpassed
            with pytest.raises(ValueError):
                llm.run(
                    ["A too long prompt. " * 1024],
                    max_new_tokens=25,
                    temperature=0.0,
                    top_p=0.0,
                    n=1,
                    repetition_penalty=None,
                    logit_bias=None,
                    batch_size=1,
                )

    @pytest.mark.skipif(
        sys.platform == "darwin", reason="instable on macOS/M2 (floating point diffs)"
    )
    def test_run(self, create_datadreamer):
        with create_datadreamer():
            # Test encoder_decoder model
            llm = HFTransformers("google/flan-t5-small")
            generated_texts = llm.run(
                [
                    "Question: What color is the sky?\nAnswer:",
                    "Question: What color are trees?\nAnswer:",
                ],
                max_new_tokens=25,
                temperature=0.0,
                top_p=0.0,
                n=2,
                stop="Question:",
                repetition_penalty=None,
                logit_bias=None,
                batch_size=2,
            )
            assert generated_texts == [["blue"] * 2, ["green"] * 2]

            # Test return_generator
            generated_texts = llm.run(
                [
                    "Question: What color is the sky?\nAnswer:",
                    "Question: What color are trees?\nAnswer:",
                ],
                max_new_tokens=25,
                temperature=0.0,
                top_p=0.0,
                n=2,
                stop="Question:",
                repetition_penalty=None,
                logit_bias=None,
                batch_size=2,
                return_generator=True,
            )
            assert isinstance(generated_texts, GeneratorType)
            assert list(generated_texts) == [["blue"] * 2, ["green"] * 2]

            # Test temperature=0.7 and seed
            generated_texts = llm.run(
                [
                    "Question: What color is the sky?\nAnswer:",
                    "Question: What color are trees?\nAnswer:",
                ],
                max_new_tokens=25,
                temperature=0.7,
                top_p=1.0,
                n=2,
                stop="Question:",
                repetition_penalty=None,
                logit_bias=None,
                batch_size=2,
                seed=42,
            )
            assert generated_texts == [["color", "blue"], ["yellow", "purple"]]

            # Test decoder-only model
            llm = HFTransformers("gpt2")
            generated_texts = llm.run(
                [
                    "Question: What color is the sky?\nAnswer:",
                    "Question: What color are trees?\nAnswer:",
                ],
                max_new_tokens=25,
                temperature=0.3,
                top_p=1.0,
                n=2,
                stop=["Question:", "Answer:", ". ", "\n"],
                repetition_penalty=None,
                logit_bias=None,
                batch_size=2,
                seed=42,
            )
            assert generated_texts == [
                [
                    "The sky is a color that is defined by the color of the sky.",
                    "The sky is the color of the night.",
                ],
                [
                    "They are the color of the ground.",
                    "The color of the trees is the color of the tree.",
                ],
            ]

            # Test logit_bias
            generated_texts = llm.run(
                [
                    "Question: What color is the sky?\nAnswer:",
                    "Question: What color are trees?\nAnswer:",
                ],
                max_new_tokens=25,
                temperature=0.3,
                top_p=1.0,
                n=1,
                stop=["Question:", "Answer:", ". ", "\n"],
                repetition_penalty=1.2,
                logit_bias={5297: 100},  # bias towards producing the token Yes
                batch_size=2,
                seed=42,
            )
            assert generated_texts == ["Yes" * 25] * 2

            # Test unload model
            assert "model" in llm.__dict__ and "tokenizer" in llm.__dict__
            llm.unload_model()
            assert "model" not in llm.__dict__ and "tokenizer" not in llm.__dict__

    def test_adapter_metadata(self, create_datadreamer):
        with create_datadreamer():
            # Load LLaMa
            llm = HFTransformers("huggyllama/llama-65b")

            # Test no PEFT citation
            assert isinstance(llm.citation, list)
            assert len(llm.citation) == 1

            # Load LLaMa + Guanaco adapter
            llm = HFTransformers(
                "huggyllama/llama-65b", adapter_name="timdettmers/guanaco-65b"
            )

            # Test PEFT + Guanaco gets added as citations
            assert isinstance(llm.citation, list)
            assert len(llm.citation) == 3

    def test_run_adapter(self, create_datadreamer):
        with create_datadreamer():
            # Generate texts without an adapter
            llm = HFTransformers(
                "gpt2", chat_prompt_template=CHAT_PROMPT_TEMPLATES["alpaca"]
            )
            generated_texts_without_adapter = llm.run(
                ["Give three tips for staying healthy."],
                max_new_tokens=25,
                temperature=1.0,
                top_p=0.0,
                n=1,
                stop=["###"],
                repetition_penalty=None,
                logit_bias=None,
                batch_size=2,
                seed=42,
            )

            # Load GPT + Alpaca adapter
            llm = HFTransformers(
                "gpt2",
                adapter_name="qblocks/gpt2_alpaca-lora",
                chat_prompt_template=CHAT_PROMPT_TEMPLATES["alpaca"],
            )

            # Generate texts with an adapter
            generated_texts_with_adapter = llm.run(
                ["Give three tips for staying healthy."],
                max_new_tokens=25,
                temperature=1.0,
                top_p=0.0,
                n=1,
                stop=["###"],
                repetition_penalty=None,
                logit_bias=None,
                batch_size=2,
                seed=42,
            )

            # Ensure the adapter was used
            with ignore_transformers_warnings():
                from peft.peft_model import PeftModelForCausalLM

            assert isinstance(get_orig_model(llm.model), PeftModelForCausalLM)

            # Compare texts with an adapter and without adapter
            assert generated_texts_without_adapter != generated_texts_with_adapter
            assert isinstance(generated_texts_without_adapter, list)
            assert isinstance(generated_texts_with_adapter, list)
            assert (
                generated_texts_without_adapter[0]
                == "Write a response that is appropriate for your body."
            )
            assert (
                generated_texts_with_adapter[0]
                == "1. Get healthy.\n2. Get healthy in the gym.\n3. Get healthy in the gym.\n4"
            )

    @pytest.mark.skipif(
        "HUGGING_FACE_HUB_TOKEN" not in os.environ, reason="requires HF Hub token"
    )
    def test_popular_chat_model(self, create_datadreamer, mocker):
        def _run_batch_mocked(*args, **kwargs):
            return [f"Response to: {p}" for p in kwargs["inputs"]]

        with create_datadreamer():
            llm = HFTransformers("meta-llama/Llama-2-7b-chat-hf")
            mocker.patch.object(llm, "_run_batch", side_effect=_run_batch_mocked)

            # Check LLaMa 2 chat prompt template is applied
            llm.run(
                ["Who was the first president?"],
                max_new_tokens=250,
                temperature=0.0,
                top_p=0.0,
                n=1,
                batch_size=6,
                batch_scheduler_buffer_size=20,
            )
            assert (
                llm._run_batch.call_args_list[0].kwargs["inputs"]  # type: ignore[attr-defined]
                == [
                    "[INST] <<SYS>>\nYou are a helpful assistant.\n<</SYS>>"
                    "\n\nWho was the first president?"
                    " [/INST] "
                ]
            )

            # Verify BOS tokens are added by the tokenizer
            assert llm.tokenizer.encode("a") == [1, 263]
            assert llm.tokenizer.encode("b") == [1, 289]

            # Verify format_prompt accounts for chat prompt template tokens
            assert llm._chat_prompt_template_token_count == 27
            assert (
                llm.count_tokens(
                    llm.format_prompt(in_context_examples=[str(i) for i in range(1500)])
                )
                + llm._chat_prompt_template_token_count
            ) == 4093

    def test_non_popular_chat_model(self, create_datadreamer, mocker):
        def _run_batch_mocked(*args, **kwargs):
            return [f"Response to: {p}" for p in kwargs["inputs"]]

        with create_datadreamer():
            llm = HFTransformers("vicgalle/gpt2-alpaca")
            mocker.patch.object(llm, "_run_batch", side_effect=_run_batch_mocked)

            # Check Alpaca chat prompt template is applied
            llm.run(
                ["Who was the first president?"],
                max_new_tokens=250,
                temperature=0.0,
                top_p=0.0,
                n=1,
                batch_size=6,
                batch_scheduler_buffer_size=20,
            )
            assert (
                llm._run_batch.call_args_list[0].kwargs["inputs"]  # type: ignore[attr-defined]
                == [
                    "Below is an instruction that describes a task. Write a response that"
                    " appropriately completes the request.\n\n### Instruction:"
                    "\nWho was the first president?\n\n### Response:\n"
                ]
            )


class TestCTransformers:
    def test_init(self, create_datadreamer):
        with create_datadreamer():
            with pytest.raises(ValueError):
                llm = CTransformers(
                    "TheBloke/CodeLlama-7B-GGUF",
                    system_prompt="You are a helpful assistant.",
                    max_context_length=4096,
                )
            with pytest.warns(UserWarning):
                llm = CTransformers("TheBloke/Llama-2-7b-Chat-GGUF")
                assert (
                    llm.display_name == "CTransformers (TheBloke/Llama-2-7b-Chat-GGUF)"
                )
            db_path = os.path.join(
                DataDreamer.get_output_folder_path(),
                ".cache",
                "CTransformers_TheBloke-Llama-2-7b-Chat-GGUF.db",
            )
            cache, _ = llm.cache_and_lock  # type: ignore[misc]
            assert os.path.exists(db_path)
            llm = CTransformers(
                "TheBloke/Llama-2-7b-Chat-GGUF",
                model_type="llama",
                max_context_length=4096,
                revision="ad37d4910ba009a69bb41de44942056d635214ab",
                local_files_only=False,
            )
            db_path = os.path.join(
                DataDreamer.get_output_folder_path(),
                ".cache",
                "CTransformers_TheBloke-Llama-2-7b-Chat-GGUF_4096_"
                "ad37d4910ba009a69bb41de44942056d635214ab_105f1de64ad82993.db",
            )
            cache, _ = llm.cache_and_lock  # type: ignore[misc]
            assert os.path.exists(db_path)
            llm = CTransformers(
                "TheBloke/Llama-2-7b-Chat-GGUF",
                model_file="llama-2-7b-chat.Q6_K.gguf",
                max_context_length=4096,
            )
            db_path = os.path.join(
                DataDreamer.get_output_folder_path(),
                ".cache",
                "CTransformers_TheBloke-Llama-2-7b-Chat-GGUF_4096_a2b3995da306d24e.db",
            )
            cache, _ = llm.cache_and_lock  # type: ignore[misc]
            assert os.path.exists(db_path)

    def test_metadata(self, create_datadreamer):
        with create_datadreamer():
            llm = CTransformers(
                "TheBloke/Llama-2-7b-Chat-GGUF",
                revision="ad37d4910ba009a69bb41de44942056d635214ab",
                max_context_length=4096,
            )
            assert (
                llm.model_card == "https://huggingface.co/TheBloke/Llama-2-7b-Chat-GGUF"
            )
            assert llm.license == (
                "https://huggingface.co/TheBloke/Llama-2-7b-Chat-GGUF/blob/"
                "ad37d4910ba009a69bb41de44942056d635214ab/LICENSE.txt"
            )
            assert isinstance(llm.citation, list)
            assert len(llm.citation) == 1

    def test_count_tokens(self, create_datadreamer):
        with create_datadreamer():
            llm = CTransformers(
                "TheBloke/Llama-2-7b-Chat-GGUF", max_context_length=4096
            )

            assert llm.count_tokens("This is a test.") == 6

            # Verify BOS tokens are added by the tokenizer
            assert llm.tokenizer.encode("a") == [1, 263]
            assert llm.tokenizer.encode("b") == [1, 289]

    def test_get_max_context_length(self, create_datadreamer):
        with create_datadreamer():
            with pytest.warns(UserWarning):
                llm = CTransformers(
                    "TheBloke/Llama-2-7b-Chat-GGUF", chat_prompt_template=None
                )
            assert llm.get_max_context_length(max_new_tokens=0) == 512
            llm = CTransformers(
                "TheBloke/Llama-2-7b-Chat-GGUF",
                max_context_length=4096,
                chat_prompt_template=None,
            )
            assert llm.get_max_context_length(max_new_tokens=0) == 4096

    def test_run(self, create_datadreamer, mocker):
        with create_datadreamer():
            # Simple test
            llm = CTransformers(
                "TheBloke/Llama-2-7b-Chat-GGUF",
                model_file="llama-2-7b-chat.Q2_K.gguf",
                max_context_length=4096,
                system_prompt="You are a helpful assistant.",
                threads=multiprocessing.cpu_count() // 2,
            )
            assert not llm._is_encoder_decoder

            # We mock this the responses to make this test run faster,
            # but for debugging, the mock can be commented out, and the test
            # should still work
            def _run_batch_mocked(*args, **kwargs):
                if llm._run_batch.call_count == 2:  # type: ignore[attr-defined]
                    return [
                        "Oh, that's a great question! *adjusts glasses* The sky"
                        " is... (checks notes)"
                    ]
                elif llm._run_batch.call_count == 1:  # type: ignore[attr-defined]
                    return [
                        "Ah, a great question! Trees are actually green!"
                        " 🌲🌿 They have green leaves"
                    ]
                elif llm._run_batch.call_count == 4:  # type: ignore[attr-defined]
                    return [
                        [
                            "Oh, that's a great question! 😊 The sky is actually blue! 🌊",
                            "Ah, a simple question! The sky is... (checks sky map) Oh, the"
                            " sky is blue today!",
                        ]
                    ]
                elif llm._run_batch.call_count == 3:  # type: ignore[attr-defined]
                    return [
                        [
                            "Trees are, of course, green! 🌲🌿 They contain chlorophyll,",
                            "Ah, a simple question! Trees are green! 🌲🌿 They are a vital part",
                        ]
                    ]

            mocker.patch.object(llm, "_run_batch", side_effect=_run_batch_mocked)

            # Simple test
            generated_texts = llm.run(
                ["What color is the sky?", "What color are trees?"],
                max_new_tokens=25,
                temperature=1.0,
                top_p=0.0,
                n=1,
                repetition_penalty=None,
                logit_bias=None,
                batch_size=10,
            )
            assert generated_texts == [
                "Oh, that's a great question! *adjusts glasses* The sky is... (checks notes)",
                "Ah, a great question! Trees are actually green! 🌲🌿 They have green leaves",
            ]

            # Test top_p and seed parameters
            generated_texts = llm.run(
                ["What color is the sky?", "What color are trees?"],
                max_new_tokens=25,
                temperature=1.0,
                top_p=1.0,
                n=2,
                repetition_penalty=None,
                logit_bias=None,
                batch_size=10,
                seed=42,
            )
            assert generated_texts == [
                [
                    "Oh, that's a great question! 😊 The sky is actually blue! 🌊",
                    "Ah, a simple question! The sky is... (checks sky map) Oh, the"
                    " sky is blue today!",
                ],
                [
                    "Trees are, of course, green! 🌲🌿 They contain chlorophyll,",
                    "Ah, a simple question! Trees are green! 🌲🌿 They are a vital part",
                ],
            ]

            # Test unload model
            assert "model" in llm.__dict__ and "tokenizer" in llm.__dict__
            llm.unload_model()
            assert "model" not in llm.__dict__ and "tokenizer" not in llm.__dict__


class TestParallelLLM:
    def test_init(self, create_datadreamer):
        with create_datadreamer():
            with pytest.raises(AssertionError):
                ParallelLLM()
            with pytest.raises(AssertionError):
                ParallelLLM("foo")  # type: ignore[arg-type]
            with pytest.raises(ValueError):
                llm_1 = HFTransformers("google/flan-t5-small")
                llm_2 = HFTransformers("gpt2")
                ParallelLLM(llm_1, llm_2)
            with pytest.raises(ValueError):
                llm = HFTransformers("google/flan-t5-small")
                ParallelLLM(llm, llm)

            # Simple test
            llm_1 = HFTransformers("gpt2")
            llm_2 = HFTransformers("gpt2")
            parallel_llm = ParallelLLM(llm_1, llm_2)
            assert parallel_llm.display_name == llm_1.display_name
            assert parallel_llm.version == llm_1.version

            # Unload model
            parallel_llm.reset_adaptive_batch_sizing()
            parallel_llm.unload_model()

    def test_parallel_llm_lock(self, create_datadreamer):
        with create_datadreamer():
            llm = ParallelLLM(
                OpenAI("gpt-3.5-turbo-instruct"), OpenAI("gpt-3.5-turbo-instruct")
            )
            with ParallelLLM(
                OpenAI("gpt-3.5-turbo-instruct"), OpenAI("gpt-3.5-turbo-instruct")
            ):
                pass

            with ParallelLLM(
                OpenAI("gpt-3.5-turbo-instruct"), OpenAI("gpt-3.5-turbo-instruct")
            ):
                with ParallelLLM(
                    OpenAI("gpt-3.5-turbo-instruct"), OpenAI("gpt-3.5-turbo-instruct")
                ):
                    pass

            def acquire_lock(llm):
                with llm:
                    sleep(0.5)

            pool = ThreadPoolExecutor(max_workers=2)
            with pytest.raises(RuntimeError):
                list(pool.map(acquire_lock, [llm, llm]))

    def test_metadata(self, create_datadreamer):
        # Simple test
        llm_1 = HFTransformers("gpt2")
        llm_2 = HFTransformers("gpt2")
        parallel_llm = ParallelLLM(llm_1, llm_2)
        assert parallel_llm.model_card == llm_1.model_card
        assert parallel_llm.license == llm_1.license
        assert parallel_llm.citation == llm_1.citation

    def test_count_tokens(self, create_datadreamer):
        with create_datadreamer():
            llm_1 = HFTransformers("gpt2")
            llm_2 = HFTransformers("gpt2")
            parallel_llm = ParallelLLM(llm_1, llm_2)
            assert parallel_llm.count_tokens("This is a test.") == llm_1.count_tokens(
                "This is a test."
            )

    def test_get_max_context_length(self, create_datadreamer):
        with create_datadreamer():
            llm_1 = HFTransformers("gpt2")
            llm_2 = HFTransformers("gpt2")
            parallel_llm = ParallelLLM(llm_1, llm_2)
            assert parallel_llm.get_max_context_length(
                max_new_tokens=0
            ) == llm_1.get_max_context_length(max_new_tokens=0)

    def test_format_prompt(self, create_datadreamer):
        with create_datadreamer():
            llm_1 = HFTransformers("gpt2")
            llm_2 = HFTransformers("gpt2")
            parallel_llm = ParallelLLM(llm_1, llm_2)
            assert parallel_llm.format_prompt(
                beg_instruction="Beg",
                in_context_examples=["1", "2"],
                end_instruction="End",
            ) == llm_1.format_prompt(
                beg_instruction="Beg",
                in_context_examples=["1", "2"],
                end_instruction="End",
            )

    def test_run_empty(self, create_datadreamer):
        with create_datadreamer():
            llm_1 = HFTransformers("gpt2")
            llm_2 = HFTransformers("gpt2")
            parallel_llm = ParallelLLM(llm_1, llm_2)
            assert (
                parallel_llm.run(
                    [],
                    max_new_tokens=25,
                    temperature=0.0,
                    top_p=0.0,
                    n=2,
                    repetition_penalty=None,
                    logit_bias=None,
                    batch_size=2,
                )
                == []
            )

    def test_run_small(self, create_datadreamer, mocker):
        prompt_1 = "This is a long prompt."
        prompt_2 = "Short prompt."

        def _run_batch_mocked(**kwargs):
            return [[f"Response to: {p}"] * kwargs["n"] for p in kwargs["inputs"]]

        with create_datadreamer():
            llm_1 = HFTransformers("gpt2")
            llm_2 = HFTransformers("gpt2")
            llm_3 = HFTransformers("gpt2")
            mocker.patch.object(llm_1, "_run_batch", side_effect=_run_batch_mocked)
            mocker.patch.object(llm_2, "_run_batch", side_effect=_run_batch_mocked)
            mocker.patch.object(llm_3, "_run_batch", side_effect=_run_batch_mocked)
            parallel_llm = ParallelLLM(llm_1, llm_2, llm_3)
            generated_texts = parallel_llm.run(
                [prompt_1, prompt_2],
                max_new_tokens=25,
                temperature=0.0,
                top_p=0.0,
                n=2,
                repetition_penalty=None,
                logit_bias=None,
                batch_size=2,
            )
            assert generated_texts == [
                [f"Response to: {prompt_1}"] * 2,
                [f"Response to: {prompt_2}"] * 2,
            ]
            assert llm_1._run_batch.call_args_list[0].kwargs["inputs"] == [  # type: ignore[attr-defined]
                prompt_2
            ]
            assert llm_2._run_batch.call_args_list[0].kwargs["inputs"] == [  # type: ignore[attr-defined]
                prompt_1
            ]
            assert llm_3._run_batch.call_count == 0  # type: ignore[attr-defined]

    def test_run_return_generator(self, create_datadreamer, mocker):
        prompt_1 = "This is a long prompt."
        prompt_2 = "Short prompt."

        def _run_batch_mocked(**kwargs):
            return [[f"Response to: {p}"] * kwargs["n"] for p in kwargs["inputs"]]

        with create_datadreamer():
            llm_1 = HFTransformers("gpt2")
            llm_2 = HFTransformers("gpt2")
            llm_3 = HFTransformers("gpt2")
            mocker.patch.object(llm_1, "_run_batch", side_effect=_run_batch_mocked)
            mocker.patch.object(llm_2, "_run_batch", side_effect=_run_batch_mocked)
            mocker.patch.object(llm_3, "_run_batch", side_effect=_run_batch_mocked)
            parallel_llm = ParallelLLM(llm_1, llm_2, llm_3)
            generated_texts = parallel_llm.run(
                [prompt_1, prompt_2],
                max_new_tokens=25,
                temperature=0.0,
                top_p=0.0,
                n=2,
                repetition_penalty=None,
                logit_bias=None,
                batch_size=2,
                return_generator=True,
            )
            assert isinstance(generated_texts, GeneratorType)
            assert list(generated_texts) == [
                [f"Response to: {prompt_1}"] * 2,
                [f"Response to: {prompt_2}"] * 2,
            ]
            assert llm_1._run_batch.call_args_list[0].kwargs["inputs"] == [  # type: ignore[attr-defined]
                prompt_2
            ]
            assert llm_2._run_batch.call_args_list[0].kwargs["inputs"] == [  # type: ignore[attr-defined]
                prompt_1
            ]
            assert llm_3._run_batch.call_count == 0  # type: ignore[attr-defined]

    def test_run_large(self, create_datadreamer, mocker, caplog):
        r = Random(42)
        prompt_1 = "This is a very very very very very very very very long prompt."
        prompt_2 = "This is a medium prompt."
        prompt_3 = "Short prompt."
        prompts = [prompt_1, prompt_2, prompt_3] * 60
        r.shuffle(prompts)

        def _run_batch_mocked(**kwargs):
            return [[f"Response to: {p}"] * kwargs["n"] for p in kwargs["inputs"]]

        with create_datadreamer():
            llm_1 = HFTransformers("gpt2")
            llm_2 = HFTransformers("gpt2")
            llm_3 = HFTransformers("gpt2")
            mocker.patch.object(llm_1, "_run_batch", side_effect=_run_batch_mocked)
            mocker.patch.object(llm_2, "_run_batch", side_effect=_run_batch_mocked)
            mocker.patch.object(llm_3, "_run_batch", side_effect=_run_batch_mocked)
            parallel_llm = ParallelLLM(llm_1, llm_2, llm_3)
            generated_texts = parallel_llm.run(
                prompts,
                max_new_tokens=25,
                temperature=0.0,
                top_p=0.0,
                n=2,
                repetition_penalty=None,
                logit_bias=None,
                batch_size=2,
                batch_scheduler_buffer_size=6,
                progress_interval=0,
                force=True,  # Don't use the cache
                return_generator=True,
            )

            # Check batches are returned from GeneratorType
            assert isinstance(generated_texts, GeneratorType)
            generated_texts = list(generated_texts)
            logs = [rec.message for rec in caplog.records]
            caplog.clear()

            # Check responses
            assert generated_texts == [
                [f"Response to: {prompt}"] * 2 for prompt in prompts
            ]

            # Test number of times each LLM is called
            assert llm_1._run_batch.call_count == 30  # type: ignore[attr-defined]
            assert llm_2._run_batch.call_count == 30  # type: ignore[attr-defined]
            assert llm_3._run_batch.call_count == 30  # type: ignore[attr-defined]

            # Test shorter prompts are executed first, but each LLM gets approximately
            # equal batches in terms of prompt length (to the extent possible)
            assert llm_1._run_batch.call_args_list[0].kwargs["inputs"] == [  # type: ignore[attr-defined]
                prompt_3,
                prompt_3,
            ]
            assert llm_2._run_batch.call_args_list[0].kwargs["inputs"] == [  # type: ignore[attr-defined]
                prompt_3,
                prompt_3,
            ]
            assert llm_3._run_batch.call_args_list[0].kwargs["inputs"] == [  # type: ignore[attr-defined]
                prompt_3,
                prompt_2,
            ]

            # Test longer prompts are executed last, but each LLM gets approximately
            # equal batches in terms of prompt length (to the extent possible)
            assert llm_1._run_batch.call_args_list[-1].kwargs["inputs"] == [  # type: ignore[attr-defined]
                prompt_2,
                prompt_1,
            ]
            assert llm_2._run_batch.call_args_list[-1].kwargs["inputs"] == [  # type: ignore[attr-defined]
                prompt_2,
                prompt_1,
            ]
            assert llm_3._run_batch.call_args_list[-1].kwargs["inputs"] == [  # type: ignore[attr-defined]
                prompt_1,
                prompt_1,
            ]

            # Test progress logs
            assert len(logs) == 121
            assert sum(["Progress: 0% 🔄" in log for log in logs]) >= 1
            assert sum(["Progress: 100% 🔄" in log for log in logs]) >= 1


class TestVLLM:
    __test__ = torch.cuda.is_available()  # Runs on GPU-only
    pydantic_version = None

    @classmethod
    def setup_class(cls):
        cls.pydantic_version = importlib.metadata.version("pydantic")
        os.system("pip3 install vllm==0.2.7")
        _reload_pydantic()

    @classmethod
    def teardown_class(cls):
        os.system(f"pip3 install pydantic=={cls.pydantic_version}")
        _reload_pydantic()

    def test_init(self, create_datadreamer):
        with create_datadreamer():
            llm = VLLM(
                "gpt2", revision="11c5a3d5811f50298f278a704980280950aedb10", device=0
            )
            db_path = os.path.join(
                DataDreamer.get_output_folder_path(),
                ".cache",
                "VLLM_gpt2_11c5a3d5811f50298f278a704980280950aedb10_None.db",
            )
            cache, _ = llm.cache_and_lock  # type: ignore[misc]
            assert os.path.exists(db_path)
            llm = VLLM("TheBloke/Mistral-7B-Instruct-v0.1-AWQ", device=0)
            assert llm.quantization == "awq"
            assert llm.chat_prompt_template is not None

    def test_metadata(self, create_datadreamer):
        with create_datadreamer():
            llm_hf = HFTransformers("gpt2")
            llm = VLLM("gpt2", device=0)

            # Check VLLM citation gets added
            assert isinstance(llm_hf.citation, list)
            assert isinstance(llm.citation, list)
            assert len(llm.citation) == len(llm_hf.citation) + 1

    def test_vllm_run(self, create_datadreamer, mocker):
        with create_datadreamer():
            llm = VLLM("gpt2", device=0)
            generated_texts = llm.run(
                [
                    "Question: What color is the sky?\nAnswer:",
                    "Question: What color are trees?\nAnswer:",
                ],
                max_new_tokens=25,
                temperature=0.0,
                top_p=1.0,
                n=1,
                repetition_penalty=None,
                logit_bias=None,
                batch_size=2,
            )
            assert isinstance(generated_texts, list)
            assert isinstance(generated_texts[0], str) and generated_texts[
                0
            ].startswith(" The sky is")
            assert isinstance(generated_texts[1], str) and generated_texts[
                1
            ].startswith(" The color")

            # Test unload model
            assert "tokenizer" in llm.__dict__
            llm.unload_model()
            assert "model" not in llm.__dict__ and "tokenizer" not in llm.__dict__

            # Test n=2
            llm = VLLM("gpt2", device=0)
            generated_texts = llm.run(
                [
                    "Question: What color is the sky?\nAnswer:",
                    "Question: What color are trees?\nAnswer:",
                ],
                max_new_tokens=25,
                temperature=0.0,
                top_p=1.0,
                n=2,
                repetition_penalty=None,
                logit_bias=None,
                batch_size=2,
            )
            assert isinstance(generated_texts, list)
            assert generated_texts[0][0].startswith(" The sky is")
            assert generated_texts[0][0] == generated_texts[0][1]
            assert generated_texts[1][0].startswith(" The color")
            assert generated_texts[1][0] == generated_texts[1][1]


class TestHFAPIEndpoint:
    def test_init(self, create_datadreamer):
        with create_datadreamer():
            llm = HFAPIEndpoint(
                "http://localhost:8080",
                "gpt2",
                revision="11c5a3d5811f50298f278a704980280950aedb10",
            )
            assert llm.display_name == "HFAPIEndpoint (http://localhost:8080)"
            db_path = os.path.join(
                DataDreamer.get_output_folder_path(),
                ".cache",
                "HFAPIEndpoint_http--localhost8080_gpt2_11c5a3d5811f50298f278a704980280950aedb10.db",
            )
            cache, _ = llm.cache_and_lock  # type: ignore[misc]
            assert os.path.exists(db_path)
            llm = HFAPIEndpoint(
                "https://api-inference.huggingface.co/models/gpt2", "gpt2"
            )
            assert llm.display_name == "HFAPIEndpoint (models/gpt2)"

    def test_metadata(self, create_datadreamer):
        with create_datadreamer():
            llm_hf = HFTransformers("gpt2")
            llm = HFAPIEndpoint("http://localhost:8080", "gpt2")
            assert llm.model_card == llm_hf.model_card
            assert llm.license == llm_hf.license
            assert llm.citation == llm_hf.citation

    def test_run(self, create_datadreamer, mocker):
        def text_generation_mocked(**kwargs):
            p = kwargs["prompt"]
            return f"Response to: {p}"

        with create_datadreamer():
            llm = HFAPIEndpoint(
                "https://api-inference.huggingface.co/models/google/flan-t5-small",
                "google/flan-t5-small",
            )
            mocker.patch.object(
                llm.client, "text_generation", side_effect=text_generation_mocked
            )
            generated_texts = llm.run(
                ["What color is the sky?", "What color are trees?"],
                max_new_tokens=25,
                temperature=0.3,
                top_p=1.0,
                n=1,
                stop=["Question:", "Answer:", ". ", "\n"],
                repetition_penalty=None,
                logit_bias=None,
                batch_size=2,
                seed=42,
            )
            assert generated_texts == [
                "Response to: What color is the sky?",
                "Response to: What color are trees?",
            ]

            # Test unload model
            assert "client" in llm.__dict__ and "tokenizer" in llm.__dict__
            llm.unload_model()
            assert "client" not in llm.__dict__ and "tokenizer" not in llm.__dict__


class TestLiteLLM:
    def test_count_tokens(self, create_datadreamer):
        with create_datadreamer():
            llm = LiteLLM("gpt-3.5-turbo")
            assert llm.count_tokens("This is a test.") == 12

    def test_get_max_context_length(self, create_datadreamer):
        with create_datadreamer():
            llm = LiteLLM("gpt-3.5-turbo-instruct")
            assert llm.get_max_context_length(max_new_tokens=0) == 4096
            llm = LiteLLM("gpt-3.5-turbo")
            assert llm.get_max_context_length(max_new_tokens=0) == 4096

    @pytest.mark.skipif(
        "OPENAI_API_KEY" not in os.environ, reason="requires OpenAI API key"
    )
    def test_run(self, create_datadreamer):
        with create_datadreamer():
            llm = LiteLLM("gpt-3.5-turbo-instruct")

            # Test Completion model
            generated_texts = llm.run(
                [
                    "Question: What color is the sky?\nAnswer:",
                    "Question: What color are trees?\nAnswer:",
                ],
                max_new_tokens=10,
                temperature=0.0,
                top_p=1.0,
                n=1,
                stop=["Question:"],
                repetition_penalty=None,
                logit_bias=None,
                batch_size=2,
            )
            assert generated_texts == [OPENAI_SKY_ANSWER, OPENAI_TREE_ANSWER]

            # Test return_generator
            generated_texts = llm.run(
                [
                    "Question: What color is the sky?\nAnswer:",
                    "Question: What color are trees?\nAnswer:",
                ],
                max_new_tokens=10,
                temperature=0.0,
                top_p=1.0,
                n=2,
                stop=["Question:"],
                repetition_penalty=None,
                logit_bias=None,
                batch_size=2,
                return_generator=True,
            )
            assert isinstance(generated_texts, GeneratorType)
            assert list(generated_texts) == [
                [OPENAI_SKY_ANSWER] * 2,
                [OPENAI_TREE_ANSWER] * 2,
            ]

            # Test unload model
            assert "client" in llm.__dict__
            llm.unload_model()
            assert "client" not in llm.__dict__


class TestAI21:
    def test_init(self, create_datadreamer):
        with create_datadreamer():
            llm = AI21("j2-light")
            assert llm._model_name_prefix == ""
            assert llm.model_card is not None
            assert llm.license is not None

    def test_run(self, create_datadreamer):
        with create_datadreamer():
            llm = AI21("j2-light", api_key="fake-key", retry_on_fail=False)
            model_calls = []

            def litellm_logger_fn(model_call_dict):
                nonlocal model_calls
                model_calls.append(model_call_dict)

            with ignore_litellm_warnings():
                from litellm.exceptions import BadRequestError

            with pytest.raises(BadRequestError):
                llm.run(
                    ["Who was the first president?"],
                    max_new_tokens=250,
                    temperature=0.0,
                    top_p=0.0,
                    n=2,
                    stop=["\n"],
                    repetition_penalty=1.1,
                    batch_size=1,
                    logger_fn=litellm_logger_fn,
                )

            assert len(model_calls) == 2
            assert model_calls[0]["additional_args"]["complete_input_dict"] == {
                "prompt": "Who was the first president?",
                "numResults": 2,
                "maxTokens": 250,
                "temperature": 1.0,
                "topP": 0.001,
                "stopSequences": ["\n"],
                "presencePenalty": {"scale": 1.1},
            }


class TestCohere:
    def test_init(self, create_datadreamer):
        with create_datadreamer():
            llm = Cohere("command-nightly")
            assert llm._model_name_prefix == ""
            assert llm.model_card is not None
            assert llm.license is not None

    def test_run(self, create_datadreamer):
        with create_datadreamer():
            llm = Cohere("command-nightly", api_key="fake-key", retry_on_fail=False)
            model_calls = []

            def litellm_logger_fn(model_call_dict):
                nonlocal model_calls
                model_calls.append(model_call_dict)

            with ignore_litellm_warnings():
                from litellm.exceptions import AuthenticationError

            with pytest.raises(AuthenticationError):
                llm.run(
                    ["Who was the first president?"],
                    max_new_tokens=250,
                    temperature=0.0,
                    top_p=0.0,
                    n=2,
                    stop=["\n"],
                    repetition_penalty=1.1,
                    batch_size=1,
                    logger_fn=litellm_logger_fn,
                )

            assert len(model_calls) == 2
            assert model_calls[0]["additional_args"]["complete_input_dict"] == {
                "model": "command-nightly",
                "prompt": "Who was the first president?",
                "temperature": 1.0,
                "max_tokens": 250,
                "num_generations": 2,
                "p": 0.001,
                "presence_penalty": 1.1,
                "stop_sequences": ["\n"],
            }


class TestAnthropic:
    def test_init(self, create_datadreamer):
        with create_datadreamer():
            llm = Anthropic("claude-3-opus-20240229")
            assert llm._model_name_prefix == ""
            assert llm.model_card is not None
            assert llm.license is not None

    def test_run(self, create_datadreamer):
        with create_datadreamer():
            llm = Anthropic(
                "claude-3-opus-20240229", api_key="fake-key", retry_on_fail=False
            )
            model_calls = []

            def litellm_logger_fn(model_call_dict):
                nonlocal model_calls
                model_calls.append(model_call_dict)

            with ignore_litellm_warnings():
                from litellm.exceptions import AuthenticationError

            with pytest.raises(AuthenticationError):
                llm.run(
                    ["Who was the first president?"],
                    max_new_tokens=250,
                    temperature=0.0,
                    top_p=0.0,
                    n=1,
                    stop=["\n"],
                    batch_size=1,
                    logger_fn=litellm_logger_fn,
                )

            assert len(model_calls) == 2
            assert model_calls[0]["additional_args"]["complete_input_dict"] == {
                "model": "claude-3-opus-20240229",
                "messages": [
                    {"content": "Who was the first president?", "role": "user"}
                ],
                "stop_sequences": ["\n"],
                "temperature": 1.0,
                "top_p": 0.001,
                "max_tokens": 250,
            }


class TestBedrock:
    @classmethod
    def setup_class(cls):
        os.system("pip3 install boto3==1.34.7")

    @pytest.mark.order("last")
    def test_init(self, create_datadreamer):
        with create_datadreamer():
            llm = Bedrock("anthropic.claude-v2")
            assert llm._model_name_prefix == ""
            assert llm.model_card is not None
            assert llm.license is not None

    @pytest.mark.order("last")
    def test_run(self, create_datadreamer):
        with create_datadreamer():
            llm = Bedrock(
                "anthropic.claude-v2",
                api_key="fake-key",
                aws_region_name="us-east-1",
                retry_on_fail=False,
            )
            model_calls = []

            def litellm_logger_fn(model_call_dict):
                nonlocal model_calls
                model_calls.append(model_call_dict)

            with ignore_litellm_warnings():
                from litellm.exceptions import (
                    AuthenticationError,
                    PermissionDeniedError,
                    ServiceUnavailableError,
                )

            with pytest.raises(
                (AuthenticationError, PermissionDeniedError, ServiceUnavailableError)
            ):
                llm.run(
                    ["Who was the first president?"],
                    max_new_tokens=250,
                    temperature=0.0,
                    top_p=0.0,
                    n=1,
                    stop=["\n"],
                    batch_size=1,
                    logger_fn=litellm_logger_fn,
                )

            assert len(model_calls) == 2
            assert json.loads(
                model_calls[0]["additional_args"]["complete_input_dict"]
            ) == {
                "prompt": "\n\nHuman: Who was the first president?\n\nAssistant: ",
                "max_tokens_to_sample": 250,
                "temperature": 1.0,
                "top_p": 0.001,
                "stop_sequences": ["\n"],
            }


class TestPaLM:
    @classmethod
    def setup_class(cls):
        os.system("pip3 install google-generativeai==0.2.2")

    @pytest.mark.order("last")
    def test_init(self, create_datadreamer):
        with create_datadreamer():
            llm = PaLM("chat-bison-001")
            assert llm._model_name_prefix == "palm/"
            assert llm.model_card is not None
            assert llm.license is not None
            assert isinstance(llm.citation, list)
            assert len(llm.citation) > 0

    @pytest.mark.order("last")
    def test_run(self, create_datadreamer):
        with create_datadreamer():
            llm = PaLM("chat-bison-001", api_key="fake-key", retry_on_fail=False)
            model_calls = []

            def litellm_logger_fn(model_call_dict):
                nonlocal model_calls
                model_calls.append(model_call_dict)

            with ignore_litellm_warnings():
                from litellm.exceptions import APIConnectionError

            with pytest.raises(APIConnectionError):
                llm.run(
                    ["Who was the first president?"],
                    max_new_tokens=250,
                    temperature=0.0,
                    top_p=0.0,
                    n=2,
                    stop=["\n"],
                    batch_size=1,
                    logger_fn=litellm_logger_fn,
                )

            assert len(model_calls) == 2
            assert model_calls[0]["additional_args"]["complete_input_dict"] == {
                "inference_params": {
                    "candidate_count": 2,
                    "max_output_tokens": 250,
                    "stop_sequences": ["\n"],
                    "temperature": 1.0,
                    "top_p": 0.001,
                }
            }


class TestVertexAI:
    @classmethod
    def setup_class(cls):
        os.system("pip3 install google-cloud-aiplatform==1.38.1")

    @pytest.mark.order("last")
    def test_init(self, create_datadreamer):
        with create_datadreamer():
            llm = VertexAI("chat-bison")
            assert llm._model_name_prefix == ""
            assert llm.model_card is not None
            assert llm.license is not None
            assert isinstance(llm.citation, list)
            assert len(llm.citation) > 0

    @pytest.mark.order("last")
    def test_run(self, create_datadreamer):
        with create_datadreamer():
            llm = VertexAI(
                "chat-bison",
                vertex_project="project",
                vertex_location="location",
                retry_on_fail=False,
            )
            model_calls = []

            def litellm_logger_fn(model_call_dict):
                nonlocal model_calls
                model_calls.append(model_call_dict)

            with ignore_litellm_warnings():
                from litellm.exceptions import APIError, BadRequestError, RateLimitError

            with pytest.raises((BadRequestError, APIError, RateLimitError)):
                llm.run(
                    ["Who was the first president?"],
                    max_new_tokens=250,
                    temperature=0.0,
                    top_p=0.0,
                    n=1,
                    stop=None,
                    batch_size=1,
                    logger_fn=litellm_logger_fn,
                )

            assert len(model_calls) == 1


class TestTogether:
    pydantic_version = None

    @classmethod
    def setup_class(cls):
        cls.pydantic_version = importlib.metadata.version("pydantic")
        os.system("pip3 install together==0.2.10")
        _reload_pydantic()

    @classmethod
    def teardown_class(cls):
        os.system(f"pip3 install pydantic=={cls.pydantic_version}")
        _reload_pydantic()

    @pytest.mark.order("last")
    def test_warnings(self, create_datadreamer):
        with create_datadreamer():
            db_path = os.path.join(
                DataDreamer.get_output_folder_path(),
                ".cache",
                "Together_nonexistant_1024_825602445ddf0f93.db",
            )
            llm = Together("nonexistant")
            with pytest.warns(UserWarning):
                cache, _ = llm.cache_and_lock  # type: ignore[misc]
            assert os.path.exists(db_path)
            db_path = os.path.join(
                DataDreamer.get_output_folder_path(),
                ".cache",
                "Together_nonexistant_2048_706c00b981a63df2.db",
            )
            llm = Together(
                "nonexistant",
                tokenizer_model_name="hf-internal-testing/llama-tokenizer",
                tokenizer_revision="99eceeba6e8289bee767f0771166b5917e70e470",
            )
            with pytest.warns(UserWarning):
                cache, _ = llm.cache_and_lock  # type: ignore[misc]
            assert os.path.exists(db_path)
            db_path = os.path.join(
                DataDreamer.get_output_folder_path(),
                ".cache",
                "Together_nonexistant_4096_825602445ddf0f93.db",
            )
            llm = Together("nonexistant", max_context_length=4096)
            with pytest.warns(UserWarning):
                cache, _ = llm.cache_and_lock  # type: ignore[misc]
            assert os.path.exists(db_path)
            db_path = os.path.join(
                DataDreamer.get_output_folder_path(),
                ".cache",
                "Together_nonexistant_4096_706c00b981a63df2.db",
            )
            llm = Together(
                "nonexistant",
                tokenizer_model_name="hf-internal-testing/llama-tokenizer",
                tokenizer_revision="99eceeba6e8289bee767f0771166b5917e70e470",
                max_context_length=4096,
            )
            cache, _ = llm.cache_and_lock  # type: ignore[misc]
            assert os.path.exists(db_path)

    @pytest.mark.order("last")
    def test_init(self, create_datadreamer):
        with create_datadreamer():
            with pytest.raises(ValueError):
                llm = Together(
                    "mistralai/Mistral-7B-Instruct-v0.1",
                    tokenizer_model_name="mistralai/Mistral-7B-Instruct-v0.1",
                    system_prompt="You are a helpful assistant.",
                )
        with create_datadreamer():
            db_path = os.path.join(
                DataDreamer.get_output_folder_path(),
                ".cache",
                "Together_mistralai-Mistral-7B-Instruct-v01_32768_ae4e96586ab1c0a4.db",
            )
            llm = Together(
                "mistralai/Mistral-7B-Instruct-v0.1",
                tokenizer_model_name="mistralai/Mistral-7B-Instruct-v0.1",
            )
            cache, _ = llm.cache_and_lock  # type: ignore[misc]
            assert os.path.exists(db_path)

    @pytest.mark.order("last")
    def test_metadata(self, create_datadreamer):
        with create_datadreamer():
            llm = Together("HuggingFaceH4/starchat-alpha")
            llm_hf = HFTransformers("HuggingFaceH4/starchat-alpha")
            assert llm.model_card is not None
            assert llm.license is not None
            assert llm.citation is not None and len(llm.citation) > 0
            assert llm.model_card == llm_hf.model_card
            assert llm.license == llm_hf.license
            assert isinstance(llm.citation, list)
            assert isinstance(llm_hf.citation, list)
            assert llm.citation[0] in llm_hf.citation

    @pytest.mark.order("last")
    def test_count_tokens(self, create_datadreamer):
        with create_datadreamer():
            llm = Together("mistralai/Mistral-7B-Instruct-v0.1")
            llm_hf = HFTransformers("mistralai/Mistral-7B-Instruct-v0.1")
            assert llm.count_tokens("This is a test.") == llm_hf.count_tokens(
                "This is a test."
            )

    @pytest.mark.order("last")
    def test_get_max_context_length(self, create_datadreamer):
        with create_datadreamer():
            llm = Together("mistralai/Mistral-7B-Instruct-v0.1")
            llm_hf = HFTransformers("mistralai/Mistral-7B-Instruct-v0.1")
            assert llm.get_max_context_length(
                max_new_tokens=0
            ) == llm_hf.get_max_context_length(max_new_tokens=0)

    @pytest.mark.skipif(
        "HUGGING_FACE_HUB_TOKEN" not in os.environ, reason="requires HF Hub token"
    )
    @pytest.mark.order("last")
    def test_run(self, create_datadreamer, mocker):
        def completion_mocked(**kwargs):
            p = kwargs["prompt"]
            return {"output": {"choices": [{"text": f"Response to: {p}"}]}}

        with create_datadreamer():
            llm = Together("togethercomputer/llama-2-7b-chat")

            # Mock Complete.create()
            mocker.patch.object(
                llm.client.Complete, "create", side_effect=completion_mocked
            )

            # Simple run
            generated_texts = llm.run(
                ["What color is the sky?", "What color are trees?"],
                max_new_tokens=25,
                temperature=0.3,
                top_p=1.0,
                n=1,
                stop=[],
                repetition_penalty=None,
                logit_bias=None,
                batch_size=2,
            )
            chat_prompt_template = llm.chat_prompt_template
            assert chat_prompt_template == CHAT_PROMPT_TEMPLATES["llama_system"]
            assert llm.system_prompt is not None
            chat_prompt_template = chat_prompt_template.replace(
                "{{system_prompt}}", llm.system_prompt
            )
            assert generated_texts == [
                "Response to: "
                + chat_prompt_template.replace(
                    "{{prompt}}", "What color is the sky?"
                ).rstrip(),
                "Response to: "
                + chat_prompt_template.replace(
                    "{{prompt}}", "What color are trees?"
                ).rstrip(),
            ]

            # Test return_generator
            generated_texts_generator = llm.run(
                ["What color is the sky?", "What color are trees?"],
                max_new_tokens=25,
                temperature=0.3,
                top_p=1.0,
                n=1,
                stop=[],
                repetition_penalty=None,
                logit_bias=None,
                batch_size=2,
                return_generator=True,
            )
            assert isinstance(generated_texts_generator, GeneratorType)
            assert list(generated_texts_generator) == generated_texts

            # Test unload model
            assert "client" in llm.__dict__ and "tokenizer" in llm.__dict__
            llm.unload_model()
            assert "client" not in llm.__dict__ and "tokenizer" not in llm.__dict__


class TestMistralAI:
    pydantic_version = None

    @classmethod
    def setup_class(cls):
        cls.pydantic_version = importlib.metadata.version("pydantic")
        os.system("pip3 install mistralai==0.0.8")
        _reload_pydantic()

    @classmethod
    def teardown_class(cls):
        os.system(f"pip3 install pydantic=={cls.pydantic_version}")
        _reload_pydantic()

    @pytest.mark.order("last")
    def test_init(self, create_datadreamer):
        with create_datadreamer():
            db_path = os.path.join(
                DataDreamer.get_output_folder_path(),
                ".cache",
                "MistralAI_mistral-tiny_32758_ae4e96586ab1c0a4.db",
            )
            llm = MistralAI("mistral-tiny")
            cache, _ = llm.cache_and_lock  # type: ignore[misc]
            assert os.path.exists(db_path)

    @pytest.mark.order("last")
    def test_metadata(self, create_datadreamer):
        with create_datadreamer():
            llm = MistralAI("HuggingFaceH4/starchat-alpha")
            assert llm.model_card is not None
            assert llm.license is not None
            assert isinstance(llm.citation, list)
            assert len(llm.citation) == 1

    @pytest.mark.order("last")
    def test_count_tokens(self, create_datadreamer):
        with create_datadreamer():
            llm = MistralAI("mistralai/Mistral-7B-Instruct-v0.1")
            llm_hf = HFTransformers("mistralai/Mistral-7B-Instruct-v0.1")
            assert llm.count_tokens("This is a test.") == llm_hf.count_tokens(
                "This is a test."
            )

    @pytest.mark.order("last")
    def test_get_max_context_length(self, create_datadreamer):
        with create_datadreamer():
            llm = MistralAI("mistralai/Mistral-7B-Instruct-v0.1")
            llm_hf = HFTransformers("mistralai/Mistral-7B-Instruct-v0.1")
            assert llm.get_max_context_length(max_new_tokens=0) == (
                llm_hf.get_max_context_length(max_new_tokens=0) - 10
            )

    @pytest.mark.order("last")
    def test_run(self, create_datadreamer, mocker):
        def chat_mocked(**kwargs):
            from mistralai.models.chat_completion import ChatCompletionResponse

            p = kwargs["messages"][0].content
            response = {
                "id": "cmpl-e5cc70bb28c444948073e77776eb30ef",
                "object": "chat.completion",
                "created": 1702256327,
                "model": "mistral-tiny",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": f"Response to: {p}",
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                },
            }
            return ChatCompletionResponse(**response)

        with create_datadreamer():
            llm = MistralAI("mistral-tiny")

            # Mock Complete.create()
            mocker.patch.object(llm.client, "chat", side_effect=chat_mocked)

            # Simple run
            generated_texts = llm.run(
                ["What color is the sky?", "What color are trees?"],
                max_new_tokens=25,
                temperature=0.3,
                top_p=1.0,
                n=1,
                stop=[],
                repetition_penalty=None,
                logit_bias=None,
                batch_size=2,
            )
            assert generated_texts == [
                "Response to: " + "What color is the sky?",
                "Response to: " + "What color are trees?",
            ]

            # Test return_generator
            generated_texts_generator = llm.run(
                ["What color is the sky?", "What color are trees?"],
                max_new_tokens=25,
                temperature=0.3,
                top_p=1.0,
                n=1,
                stop=[],
                repetition_penalty=None,
                logit_bias=None,
                batch_size=2,
                return_generator=True,
            )
            assert isinstance(generated_texts_generator, GeneratorType)
            assert list(generated_texts_generator) == generated_texts

            # Test unload model
            assert "client" in llm.__dict__ and "tokenizer" in llm.__dict__
            llm.unload_model()
            assert "client" not in llm.__dict__ and "tokenizer" not in llm.__dict__


class TestPetals:
    pydantic_version = None

    @classmethod
    def setup_class(cls):
        cls.pydantic_version = importlib.metadata.version("pydantic")
        os.system("pip3 install petals==2.2.0")
        os.system("pip3 install 'pydantic>=1.10,<2.0'")
        _reload_pydantic()

    @classmethod
    def teardown_class(cls):
        os.system(f"pip3 install pydantic=={cls.pydantic_version}")
        _reload_pydantic()

    @pytest.mark.xfail  # Petals network is unreliable currently
    @pytest.mark.timeout(90)
    @pytest.mark.order("last")
    def test_init(self, create_datadreamer):
        with create_datadreamer():
            llm = Petals("petals-team/StableBeluga2")
            db_path = os.path.join(
                DataDreamer.get_output_folder_path(),
                ".cache",
                "Petals_petals-team-StableBeluga2_torch.bfloat16.db",
            )
            cache, _ = llm.cache_and_lock  # type: ignore[misc]
            assert os.path.exists(db_path)
            llm = Petals(
                "huggyllama/llama-65b",
                revision="49707c5313d34d1c5a846e29cf2a2a650c22c8ee",
                trust_remote_code=True,
                dtype=torch.float16,
                adapter_name="timdettmers/guanaco-65b",
            )
            db_path = os.path.join(
                DataDreamer.get_output_folder_path(),
                ".cache",
                "Petals_huggyllama-llama-65b_timdettmers-guanaco-65b_49707c5313d34d1c5a846e29cf2a2a650c22c8ee_torch.float16.db",
            )
            cache, _ = llm.cache_and_lock  # type: ignore[misc]
            assert os.path.exists(db_path)

    @pytest.mark.order("last")
    def test_metadata(self, create_datadreamer):
        with create_datadreamer():
            llm_hf = HFTransformers("petals-team/StableBeluga2")
            llm = Petals("petals-team/StableBeluga2")

            # Check Petals citation gets added
            assert isinstance(llm_hf.citation, list)
            assert isinstance(llm.citation, list)
            assert len(llm.citation) == len(llm_hf.citation) + 1

    @pytest.mark.xfail  # Petals network is unreliable currently
    @pytest.mark.timeout(90)
    @pytest.mark.order("last")
    def test_petals_network(self, create_datadreamer):
        with create_datadreamer():
            llm = Petals("petals-team/StableBeluga2", dtype=torch.float32)
            generated_texts = llm.run(
                ["A", "B"],
                max_new_tokens=1,
                temperature=1.0,
                top_p=0.0,
                n=1,
                repetition_penalty=None,
                logit_bias=None,
                batch_size=1,
            )
            assert isinstance(generated_texts, list)
            assert len(generated_texts) == 2

    @pytest.mark.xfail  # Petals network is unreliable currently
    @pytest.mark.timeout(90)
    @pytest.mark.order("last")
    def test_run(self, create_datadreamer, mocker):
        with create_datadreamer():
            llm = Petals("petals-team/StableBeluga2")

            # We mock this the responses to make this test run faster,
            # but for debugging, the mock can be commented out, and the test
            # should still work
            def _run_batch_mocked(*args, **kwargs):
                return [
                    [
                        "Trees can come in a variety of colors, depending on the species"
                        " and the season. Generally, trees are green due"
                    ]
                    * 2,
                    [
                        "The color of the sky can vary depending on the time of day,"
                        " weather conditions, and the location. Generally, during"
                    ]
                    * 2,
                ]

            mocker.patch.object(llm, "_run_batch", side_effect=_run_batch_mocked)

            generated_texts = llm.run(
                ["What color is the sky?", "What color are trees?"],
                max_new_tokens=25,
                temperature=1.0,
                top_p=0.0,
                n=2,
                repetition_penalty=None,
                logit_bias=None,
                batch_size=2,
            )
            assert generated_texts == [
                [
                    "The color of the sky can vary depending on the time of day,"
                    " weather conditions, and the location. Generally, during"
                ]
                * 2,
                [
                    "Trees can come in a variety of colors, depending on the species"
                    " and the season. Generally, trees are green due"
                ]
                * 2,
            ]

    @pytest.mark.xfail  # Petals network is unreliable currently
    @pytest.mark.timeout(90)
    @pytest.mark.order("last")
    def test_adaptive_batch_size(self, create_datadreamer, mocker):
        from hivemind.p2p.p2p_daemon_bindings.utils import P2PHandlerError

        def _run_batch_mocked(*args, **kwargs):
            raise P2PHandlerError("Could not allocate out of memory")

        with create_datadreamer():
            llm = Petals("petals-team/StableBeluga2")
            mocker.patch.object(llm, "_run_batch", side_effect=_run_batch_mocked)

            # If OOM error is constantly thrown, we will keep trying smaller batch sizes
            # until we get to a batch size of 1, at which point the OOM error is raised
            # to the user
            with pytest.raises(P2PHandlerError):
                llm.run(
                    [
                        f"{i} " + " ".join([f"test_{num_words}"] * num_words)
                        for i, num_words in enumerate(([19] * 6))
                    ],
                    max_new_tokens=25,
                    temperature=0.0,
                    top_p=0.0,
                    n=1,
                    batch_size=6,
                    batch_scheduler_buffer_size=20,
                )
            assert llm._run_batch.call_count == 6  # type: ignore[attr-defined]
