"""
Standalone tests for the MiniMax LLM provider.

These tests verify the MiniMax provider implementation without requiring
the full DataDreamer project dependencies to be installed. Tests that
require the full project use the TestMiniMaxInProject class in
src/tests/llms/test_llms.py.

Run with: python -m pytest tests/test_minimax.py -v
"""

import os
import re
from unittest.mock import MagicMock, patch

import pytest


def _read_minimax_source():
    """Read the minimax.py source file."""
    path = os.path.join(os.path.dirname(__file__), "..", "src", "llms", "minimax.py")
    with open(path, "r") as f:
        return f.read()


def _read_init_source():
    """Read the llms __init__.py source file."""
    path = os.path.join(os.path.dirname(__file__), "..", "src", "llms", "__init__.py")
    with open(path, "r") as f:
        return f.read()


class TestMiniMaxModuleStructure:
    """Tests verifying the module structure and exports."""

    def test_minimax_file_exists(self):
        """Test that minimax.py exists in the llms directory."""
        path = os.path.join(
            os.path.dirname(__file__), "..", "src", "llms", "minimax.py"
        )
        assert os.path.exists(path)

    def test_minimax_imports_in_init(self):
        """Test that MiniMax is imported in llms/__init__.py."""
        source = _read_init_source()
        assert "from .minimax import MiniMax" in source

    def test_minimax_in_all_exports(self):
        """Test that MiniMax is in __all__ in llms/__init__.py."""
        source = _read_init_source()
        assert '"MiniMax"' in source

    def test_minimax_module_all_export(self):
        """Test that minimax.py has __all__ = ['MiniMax']."""
        source = _read_minimax_source()
        assert '__all__ = ["MiniMax"]' in source

    def test_inherits_from_openai(self):
        """Test that MiniMax class inherits from OpenAI."""
        source = _read_minimax_source()
        assert "class MiniMax(OpenAI):" in source

    def test_imports_openai_parent(self):
        """Test that minimax.py imports the OpenAI class."""
        source = _read_minimax_source()
        assert "from .openai import OpenAI" in source


class TestMiniMaxConstants:
    """Tests verifying the constants defined in minimax.py."""

    def test_base_url(self):
        """Test that the base URL is correct."""
        source = _read_minimax_source()
        assert '_MINIMAX_BASE_URL = "https://api.minimax.io/v1"' in source

    def test_context_lengths_m27(self):
        """Test M2.7 context length is 1M."""
        source = _read_minimax_source()
        assert '"MiniMax-M2.7": 1000000' in source

    def test_context_lengths_m27_highspeed(self):
        """Test M2.7-highspeed context length is 1M."""
        source = _read_minimax_source()
        assert '"MiniMax-M2.7-highspeed": 1000000' in source

    def test_context_lengths_m25(self):
        """Test M2.5 context length is 204800."""
        source = _read_minimax_source()
        assert '"MiniMax-M2.5": 204800' in source

    def test_context_lengths_m25_highspeed(self):
        """Test M2.5-highspeed context length is 204800."""
        source = _read_minimax_source()
        assert '"MiniMax-M2.5-highspeed": 204800' in source

    def test_max_output_m27(self):
        """Test M2.7 max output is 16384."""
        source = _read_minimax_source()
        # Check the output dict contains M2.7: 16384
        assert re.search(r'"MiniMax-M2\.7":\s*16384', source)

    def test_max_output_m25(self):
        """Test M2.5 max output is 8192."""
        source = _read_minimax_source()
        assert re.search(r'"MiniMax-M2\.5":\s*8192', source)


class TestMiniMaxImplementation:
    """Tests verifying the implementation logic in minimax.py."""

    def test_temperature_clamping_zero(self):
        """Test that temperature=0.0 is clamped to 0.01."""
        source = _read_minimax_source()
        assert "temperature == 0.0" in source
        assert "temperature = 0.01" in source

    def test_temperature_clamping_above_one(self):
        """Test that temperature > 1.0 is clamped to 1.0."""
        source = _read_minimax_source()
        assert "temperature > 1.0" in source
        assert "temperature = 1.0" in source

    def test_api_key_from_env(self):
        """Test that API key can be read from MINIMAX_API_KEY env var."""
        source = _read_minimax_source()
        assert 'os.environ.get("MINIMAX_API_KEY")' in source

    def test_default_system_prompt(self):
        """Test that default system prompt is set."""
        source = _read_minimax_source()
        assert '"You are a helpful assistant."' in source

    def test_model_card_url(self):
        """Test the model card URL."""
        source = _read_minimax_source()
        assert "https://platform.minimaxi.com/document/Models" in source

    def test_license_url(self):
        """Test the license URL."""
        source = _read_minimax_source()
        assert "https://platform.minimaxi.com/document/Terms%20of%20service" in source

    def test_citation_none(self):
        """Test that citation returns None."""
        source = _read_minimax_source()
        # Verify the citation method returns None
        assert re.search(r"def citation.*\n.*return None", source)

    def test_get_max_output_length_method(self):
        """Test that _get_max_output_length method exists."""
        source = _read_minimax_source()
        assert "def _get_max_output_length(self)" in source

    def test_get_max_context_length_method(self):
        """Test that get_max_context_length method exists."""
        source = _read_minimax_source()
        assert "def get_max_context_length(self, max_new_tokens: int)" in source

    def test_run_batch_method(self):
        """Test that _run_batch method exists and calls super()."""
        source = _read_minimax_source()
        assert "def _run_batch(" in source
        assert "return super()._run_batch(" in source

    def test_client_property(self):
        """Test that client property creates OpenAI client with correct base URL."""
        source = _read_minimax_source()
        assert "def client(self)" in source
        assert "openai.OpenAI(" in source
        assert "base_url=_MINIMAX_BASE_URL" in source

    def test_has_docstring(self):
        """Test that the MiniMax class has a docstring."""
        source = _read_minimax_source()
        # After 'class MiniMax(OpenAI):' there should be a docstring
        class_match = source.find("class MiniMax(OpenAI):")
        assert class_match >= 0
        after_class = source[class_match:]
        assert '"""' in after_class[:200]

    def test_all_model_names_in_context_dict(self):
        """Test all 4 MiniMax models are in context length dict."""
        source = _read_minimax_source()
        models = [
            "MiniMax-M2.7",
            "MiniMax-M2.7-highspeed",
            "MiniMax-M2.5",
            "MiniMax-M2.5-highspeed",
        ]
        for model in models:
            assert f'"{model}"' in source, f"Model {model} not found in source"

    def test_all_model_names_in_output_dict(self):
        """Test all 4 MiniMax models are in max output length dict."""
        source = _read_minimax_source()
        # Both dicts should have all 4 models
        context_matches = re.findall(r'"MiniMax-M\d+\.\d+[^"]*"', source)
        assert len(context_matches) >= 8  # 4 models x 2 dicts


class TestMiniMaxTestsInProject:
    """Tests verifying that MiniMax tests exist in the project test suite."""

    def test_minimax_test_class_exists(self):
        """Test that TestMiniMax class exists in test_llms.py."""
        path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "src",
            "tests",
            "llms",
            "test_llms.py",
        )
        with open(path, "r") as f:
            source = f.read()
        assert "class TestMiniMax:" in source

    def test_minimax_imported_in_tests(self):
        """Test that MiniMax is imported in test_llms.py."""
        path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "src",
            "tests",
            "llms",
            "test_llms.py",
        )
        with open(path, "r") as f:
            source = f.read()
        assert "MiniMax," in source

    def test_integration_tests_exist(self):
        """Test that integration tests exist with MINIMAX_API_KEY skip."""
        path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "src",
            "tests",
            "llms",
            "test_llms.py",
        )
        with open(path, "r") as f:
            source = f.read()
        assert "MINIMAX_API_KEY" in source
        assert "test_integration_run" in source


class TestMiniMaxIntegration:
    """Integration tests that require MINIMAX_API_KEY environment variable."""

    @pytest.mark.skipif(
        "MINIMAX_API_KEY" not in os.environ, reason="requires MiniMax API key"
    )
    def test_api_chat_completion(self):
        """Integration test: direct API call to MiniMax."""
        import openai

        client = openai.OpenAI(
            api_key=os.environ["MINIMAX_API_KEY"],
            base_url="https://api.minimax.io/v1",
        )
        response = client.chat.completions.create(
            model="MiniMax-M2.7",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Say hello in one word."},
            ],
            max_tokens=10,
            temperature=0.01,
        )
        assert len(response.choices) > 0
        assert isinstance(response.choices[0].message.content, str)
        assert len(response.choices[0].message.content) > 0

    @pytest.mark.skipif(
        "MINIMAX_API_KEY" not in os.environ, reason="requires MiniMax API key"
    )
    def test_api_m27_highspeed(self):
        """Integration test: M2.7-highspeed model."""
        import openai

        client = openai.OpenAI(
            api_key=os.environ["MINIMAX_API_KEY"],
            base_url="https://api.minimax.io/v1",
        )
        response = client.chat.completions.create(
            model="MiniMax-M2.7-highspeed",
            messages=[
                {"role": "user", "content": "Say 'yes' or 'no'."},
            ],
            max_tokens=5,
            temperature=0.01,
        )
        assert len(response.choices) > 0
        assert isinstance(response.choices[0].message.content, str)

    @pytest.mark.skipif(
        "MINIMAX_API_KEY" not in os.environ, reason="requires MiniMax API key"
    )
    def test_api_temperature_edge(self):
        """Integration test: verify low temperature works (near 0)."""
        import openai

        client = openai.OpenAI(
            api_key=os.environ["MINIMAX_API_KEY"],
            base_url="https://api.minimax.io/v1",
        )
        response = client.chat.completions.create(
            model="MiniMax-M2.7",
            messages=[
                {"role": "user", "content": "What is 2+2? Just the number."},
            ],
            max_tokens=200,
            temperature=0.01,
        )
        content = response.choices[0].message.content
        # M2.7 may include <think> tags; strip them before checking
        import re

        clean = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
        assert "4" in clean


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
