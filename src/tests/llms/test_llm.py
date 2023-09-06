import os
from math import ceil, floor

import pytest

from ... import DataDreamer
from ...llms import HFTransformers, OpenAI


class TestLLM:
    def test_cache(self, create_datadreamer):
        assert OpenAI("text-ada-001").get_cache() is None
        with create_datadreamer():
            db_path = os.path.join(
                DataDreamer.get_output_folder_path(),
                ".llm_cache",
                "OpenAI_text-ada-001_e426fca92c28fbdc.db",
            )
            db_lock_path = db_path + ".flock"
            llm = OpenAI("text-ada-001")
            cache_dict, cache_flock = llm.get_cache()  # type: ignore[misc]
            assert os.path.exists(db_path)
            with cache_flock:
                assert os.path.exists(db_lock_path)
            assert cache_dict.filename == db_path
            assert cache_dict.journal_mode == "WAL"
            assert not cache_dict.autocommit
            assert cache_flock.lock_file == db_lock_path

    def test_count_tokens(self, create_datadreamer):
        with create_datadreamer():
            llm = OpenAI("text-ada-001")
            assert llm.count_tokens("This is a test.") == 5

    def test_get_max_context_length(self, create_datadreamer):
        with create_datadreamer():
            llm = OpenAI("text-ada-001")
            assert llm.get_max_context_length(max_new_tokens=0) == 2049
            llm = OpenAI("gpt-3.5-turbo")
            assert llm.get_max_context_length(max_new_tokens=0) == 4084

    def test_get_prompt(self, create_datadreamer):
        with create_datadreamer():
            llm = OpenAI("text-ada-001")
            assert llm.get_max_context_length(max_new_tokens=0) == 2049
            assert (
                llm.get_prompt(
                    max_new_tokens=0,
                    beg_instruction="Beg",
                    in_context_examples=["1", "2"],
                    end_instruction="End",
                )
                == "Beg\n1\n2\nEnd"
            )
            assert (
                llm.get_prompt(
                    max_new_tokens=0,
                    beg_instruction=None,
                    in_context_examples=None,
                    end_instruction=None,
                )
                == ""
            )
            assert (
                llm.get_prompt(
                    max_new_tokens=0,
                    beg_instruction="Beg",
                    in_context_examples=None,
                    end_instruction="End",
                )
                == "Beg\nEnd"
            )
            assert (
                llm.get_prompt(
                    max_new_tokens=0,
                    beg_instruction=None,
                    in_context_examples=["1", "2"],
                    end_instruction="End",
                )
                == "1\n2\nEnd"
            )
            assert (
                llm.get_prompt(
                    max_new_tokens=0,
                    beg_instruction="Beg",
                    in_context_examples=["1", "2"],
                    end_instruction=None,
                )
                == "Beg\n1\n2"
            )
            assert (
                llm.get_prompt(
                    max_new_tokens=0,
                    beg_instruction=None,
                    in_context_examples=["1", "2"],
                    end_instruction=None,
                )
                == "1\n2"
            )

    def test_get_prompt_error_instruction_too_large(self):
        llm = OpenAI("text-ada-001")
        single_token = "aaaa"

        # Test just beginning instruction too large
        max_content_length = llm.get_max_context_length(max_new_tokens=0)
        llm.get_prompt(
            max_new_tokens=0, beg_instruction=single_token * (max_content_length - 1)
        )
        llm.get_prompt(
            max_new_tokens=0, beg_instruction=single_token * (max_content_length)
        )
        with pytest.raises(ValueError):
            llm.get_prompt(
                max_new_tokens=0,
                beg_instruction=single_token * (max_content_length + 1),
            )

        # Test beg + end together too large
        # We do the "- 1" in max_content_length_left to account for the sep token
        max_content_length = llm.get_max_context_length(max_new_tokens=0)
        max_content_length_left = floor(max_content_length / 2.0) - 1
        max_content_length_right = ceil(max_content_length / 2.0)
        llm.get_prompt(
            max_new_tokens=0,
            beg_instruction=single_token * (max_content_length_left - 1),
            end_instruction=single_token * max_content_length_right,
        )
        llm.get_prompt(
            max_new_tokens=0,
            beg_instruction=single_token * (max_content_length_left),
            end_instruction=single_token * max_content_length_right,
        )
        with pytest.raises(ValueError):
            llm.get_prompt(
                max_new_tokens=0,
                beg_instruction=single_token * (max_content_length_left + 1),
                end_instruction=single_token * max_content_length_right,
            )

    def test_get_prompt_error_instruction_too_large_with_in_context_examples(self):
        llm = OpenAI("text-ada-001")
        single_token = "aaaa"

        # Test just beginning instruction too large
        # We do the "- 1" in max_content_length to account for the sep token
        max_content_length = llm.get_max_context_length(max_new_tokens=0) - 1
        llm.get_prompt(
            max_new_tokens=0,
            beg_instruction=single_token * (max_content_length - 1),
            in_context_examples=[single_token],
        )
        with pytest.raises(ValueError):
            llm.get_prompt(
                max_new_tokens=0,
                beg_instruction=single_token * (max_content_length),
                in_context_examples=[single_token],
            )
        with pytest.raises(ValueError):
            llm.get_prompt(
                max_new_tokens=0,
                beg_instruction=single_token * (max_content_length + 1),
                in_context_examples=[single_token],
            )

        # Test beg + end together too large
        # We do the "- 2" in max_content_length_left to account for the sep tokens
        max_content_length = llm.get_max_context_length(max_new_tokens=0)
        max_content_length_left = floor(max_content_length / 2.0) - 2
        max_content_length_right = ceil(max_content_length / 2.0)
        llm.get_prompt(
            max_new_tokens=0,
            beg_instruction=single_token * (max_content_length_left - 1),
            in_context_examples=[single_token],
            end_instruction=single_token * max_content_length_right,
        )
        with pytest.raises(ValueError):
            llm.get_prompt(
                max_new_tokens=0,
                beg_instruction=single_token * (max_content_length_left),
                in_context_examples=[single_token],
                end_instruction=single_token * max_content_length_right,
            )
        with pytest.raises(ValueError):
            llm.get_prompt(
                max_new_tokens=0,
                beg_instruction=single_token * (max_content_length_left + 1),
                in_context_examples=[single_token],
                end_instruction=single_token * max_content_length_right,
            )

    def test_too_many_in_context_examples(self):
        llm = OpenAI("text-ada-001")
        single_token = "aaaa"

        # Test too many in-context examples gets truncated
        # We do the "- 1" in max_content_length to account for the sep token
        max_content_length = llm.get_max_context_length(max_new_tokens=0) - 1
        assert (
            llm.get_prompt(
                max_new_tokens=0,
                beg_instruction=single_token * (max_content_length - 1),
                in_context_examples=["foo", "bar"],
            )
            == (single_token * (max_content_length - 1)) + "\n" + "foo"
        )

        # Test too many in-context examples gets truncated to those that can fit
        # We do the "- 1" in max_content_length to account for the sep token
        max_content_length = llm.get_max_context_length(max_new_tokens=0) - 1
        assert (
            llm.get_prompt(
                max_new_tokens=0,
                beg_instruction=single_token * (max_content_length - 1),
                in_context_examples=["This is a longer example that won't fit", "bar"],
            )
            == (single_token * (max_content_length - 1)) + "\n" + "bar"
        )

        # Test too many in-context examples gets truncated to none
        # We do the "- 1" in max_content_length to account for the sep token
        max_content_length = llm.get_max_context_length(max_new_tokens=0) - 1
        with pytest.warns(UserWarning):
            assert llm.get_prompt(
                max_new_tokens=0,
                beg_instruction=single_token * (max_content_length - 1),
                in_context_examples=["This is a longer example that won't fit"],
            ) == (single_token * (max_content_length - 1))

    def test_run(self, create_datadreamer):
        with create_datadreamer():
            llm = OpenAI("text-ada-001")
            generated_texts = llm.run(
                [
                    "Question: What color is the sky?\nAnswer:",
                    "Question: What color are trees?\nAnswer:",
                ],
                max_new_tokens=25,
                temperature=0.0,
                top_p=1.0,
                n=2,
                stop=["\nQuestion:"],
                repetition_penalty=None,
                logit_bias=None,
                batch_size=2,
            )
            assert generated_texts == [
                ["The sky is blue."] * 2,
                [
                    "Trees are typically a color that agrees with their location"
                    " in the forest: green, brown, or red."
                ]
                * 2,
            ]
            llm = OpenAI("gpt-3.5-turbo")
            generated_texts = llm.run(
                [
                    "What color is the sky?",
                    "What color are trees?",
                ],
                max_new_tokens=25,
                temperature=0.0,
                top_p=1.0,
                n=1,
                stop=". ",
                repetition_penalty=None,
                logit_bias=None,
                batch_size=2,
            )
            assert generated_texts == [
                "The color of the sky can vary depending on the time of day"
                " and weather conditions",
                "Trees can be various shades of green, but they can also have"
                " other colors depending on the season",
            ]


class TestHFTransformers:
    def test_init(self, create_datadreamer):
        with create_datadreamer():
            llm = HFTransformers(
                "google/flan-t5-small",
                revision="2d036ee774a9cb8d7e03c9f2e78ae0a16343a9d9",
                device_map="auto",
            )
            db_path = os.path.join(
                DataDreamer.get_output_folder_path(),
                ".llm_cache",
                "HFTransformers_google-flan-t5-small_2d036ee774a9cb8d7e03c9f2e78ae0a16343a9d9_torch.float32.db",
            )
            cache_dict, _ = llm.get_cache()  # type: ignore[misc]
            assert os.path.exists(db_path)

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
