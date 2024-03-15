import importlib
import os
from types import GeneratorType, SimpleNamespace

import pytest
from flaky import flaky

from ... import DataDreamer
from ...embedders import (
    OpenAIEmbedder,
    ParallelEmbedder,
    SentenceTransformersEmbedder,
    TogetherEmbedder,
)
from ..llms.test_llms import _reload_pydantic


class TestSentenceTransformersEmbedder:
    def test_init(self, create_datadreamer):
        with create_datadreamer():
            db_path = os.path.join(
                DataDreamer.get_output_folder_path(),
                ".cache",
                "SentenceTransformersEmbedder_all-mpnet-base-v2_torch.float32.db",
            )
            model = SentenceTransformersEmbedder("all-mpnet-base-v2")
            cache, _ = model.cache_and_lock  # type: ignore[misc]
            assert os.path.exists(db_path)

    def test_metadata(self, create_datadreamer):
        with create_datadreamer():
            model = SentenceTransformersEmbedder("all-mpnet-base-v2")
            assert model.model_card is not None
            assert model.license is not None
            assert isinstance(model.citation, list)
            assert len(model.citation) == 1

    def test_run(self, create_datadreamer):
        with create_datadreamer():
            # Simple test
            model = SentenceTransformersEmbedder("all-mpnet-base-v2")
            assert model.model_max_length == 384
            assert model.dims == 768
            results = model.run(texts=["A test sentence.", "Another test sentence."])
            assert isinstance(results, list)
            assert len(results[0]) == 768
            assert results[0][0] == pytest.approx(0.042875204, 0.0001)
            assert results[1][0] == pytest.approx(0.027576972, 0.0001)

            # Simple test with instruction
            model = SentenceTransformersEmbedder("all-mpnet-base-v2")
            results = model.run(
                texts=["A test sentence.", "Another test sentence."],
                instruction="Represent this sentence: ",
            )
            assert isinstance(results, list)
            assert len(results[0]) == 768
            assert results[0][0] == pytest.approx(0.005033156, 0.0001)
            assert results[1][0] == pytest.approx(0.011253911, 0.0001)

            # Test with causal model (not specfically made to be an embedding model)
            model = SentenceTransformersEmbedder("sshleifer/tiny-gpt2")
            results = model.run(texts=["A test sentence.", "Another test sentence."])
            assert isinstance(results, list)
            assert len(results[0]) == 2

            # Test truncate
            with pytest.raises(ValueError):
                results = model.run(texts=["A test sentence." * 10000], truncate=False)

            # Test return_generator=True
            results = model.run(
                texts=["A test sentence.", "Another test sentence."],
                return_generator=True,
            )
            assert isinstance(results, GeneratorType)
            assert len(list(results)) == 2

            # Test unload model
            assert "model" in model.__dict__ and "tokenizer" in model.__dict__
            model.unload_model()
            assert "model" not in model.__dict__ and "tokenizer" not in model.__dict__

    @pytest.mark.skip(
        reason="Skipping instructor embedding, requires external library."
    )
    def test_instructor_embedding(self, create_datadreamer):
        with create_datadreamer():
            # Simple test with instruction
            model = SentenceTransformersEmbedder("hkunlp/instructor-base")
            assert isinstance(model.citation, list)
            assert len(model.citation) == 2
            results = model.run(
                texts=["A test sentence.", "Another test sentence."],
                instruction="Represent this sentence: ",
            )
            assert isinstance(results, list)
            assert len(results[0]) == 768
            assert results[0][0] == pytest.approx(-0.017620174, 0.0001)
            assert results[1][0] == pytest.approx(-0.008969681, 0.0001)


class TestParallelEmbedder:
    def test_run(self, create_datadreamer):
        with create_datadreamer():
            # Simple test
            model_1 = SentenceTransformersEmbedder("all-mpnet-base-v2")
            model_2 = SentenceTransformersEmbedder("all-mpnet-base-v2")
            parallel_model = ParallelEmbedder(model_1, model_2)
            assert parallel_model.model_max_length == 384
            assert parallel_model.dims == 768
            assert parallel_model.count_tokens("") == 2
            results = parallel_model.run(
                texts=["A test sentence.", "Another test sentence."], batch_size=1
            )
            assert isinstance(results, list)
            assert len(results[0]) == 768
            assert results[0][0] == pytest.approx(0.042875204, 0.0001)
            assert results[1][0] == pytest.approx(0.027576972, 0.0001)

            # Test return_generator=True
            results = parallel_model.run(
                texts=["A test sentence.", "Another test sentence."],
                return_generator=True,
            )
            assert isinstance(results, GeneratorType)
            assert len(list(results)) == 2

            # Unload model
            parallel_model.unload_model()


class TestOpenAIEmbedder:
    def test_init(self, create_datadreamer):
        with create_datadreamer():
            db_path = os.path.join(
                DataDreamer.get_output_folder_path(),
                ".cache",
                "OpenAIEmbedder_text-embedding-3-small_1024.db",
            )
            model = OpenAIEmbedder("text-embedding-3-small", dimensions=1024)
            cache, _ = model.cache_and_lock  # type: ignore[misc]
            assert os.path.exists(db_path)

    def test_metadata(self, create_datadreamer):
        with create_datadreamer():
            model = OpenAIEmbedder("text-embedding-3-small")
            assert model.model_card is not None
            assert model.license is not None
            assert isinstance(model.citation, list)
            assert len(model.citation) == 1

    @pytest.mark.skipif(
        "OPENAI_API_KEY" not in os.environ, reason="requires OpenAI API key"
    )
    @flaky(max_runs=3)
    def test_run(self, create_datadreamer):
        with create_datadreamer():
            # Simple test
            model = OpenAIEmbedder("text-embedding-3-small", dimensions=1024)
            assert model.model_max_length == 8191
            assert model.dims == 1024
            results = model.run(texts=["A test sentence.", "Another test sentence."])
            assert isinstance(results, list)
            assert len(results[0]) == 1024
            assert results[0][0] == pytest.approx(0.016176335513591766, 0.01)
            assert results[1][0] == pytest.approx(0.008149503730237484, 0.01)

            # Test truncate
            with pytest.raises(ValueError):
                results = model.run(texts=["A test sentence." * 10000], truncate=False)

            # Test return_generator=True
            results = model.run(
                texts=["A test sentence.", "Another test sentence."],
                return_generator=True,
            )
            assert isinstance(results, GeneratorType)
            assert len(list(results)) == 2

            # Test unload model
            assert "client" in model.__dict__ and "tokenizer" in model.__dict__
            model.unload_model()
            assert "client" not in model.__dict__ and "tokenizer" not in model.__dict__


class TestTogetherEmbedder:
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
    def test_init(self, create_datadreamer):
        with create_datadreamer():
            db_path = os.path.join(
                DataDreamer.get_output_folder_path(),
                ".cache",
                "TogetherEmbedder_togethercomputer-m2-bert-80M-2k-retrieval_2048_1a8c373a8b9f2a4e.db",
            )
            model = TogetherEmbedder(
                "togethercomputer/m2-bert-80M-2k-retrieval",
                tokenizer_trust_remote_code=True,
                max_context_length=2048,
            )
            cache, _ = model.cache_and_lock  # type: ignore[misc]
            assert os.path.exists(db_path)

    @pytest.mark.order("last")
    def test_metadata(self, create_datadreamer):
        with create_datadreamer():
            model = TogetherEmbedder(
                "togethercomputer/m2-bert-80M-2k-retrieval",
                tokenizer_trust_remote_code=True,
                max_context_length=2048,
            )
            assert model.model_card is not None
            assert model.license is not None
            assert isinstance(model.citation, list)
            assert len(model.citation) == 1

    @flaky(max_runs=3)
    @pytest.mark.order("last")
    def test_run(self, create_datadreamer, mocker):
        with create_datadreamer():
            # Simple test
            model = TogetherEmbedder(
                "togethercomputer/m2-bert-80M-2k-retrieval",
                tokenizer_trust_remote_code=True,
                max_context_length=2048,
            )
            assert model.model_max_length == 2048
            assert model.dims == 768

            def _create_embeddings_mocked(*args, **kwargs):
                return SimpleNamespace(
                    data=[
                        SimpleNamespace(embedding=[0.1] * 768),
                        SimpleNamespace(embedding=[0.2] * 768),
                    ]
                )

            mocker.patch.object(
                model.client.Embeddings, "create", _create_embeddings_mocked
            )
            results = model.run(texts=["A test sentence.", "Another test sentence."])
            assert isinstance(results, list)
            assert len(results) == 2
            assert len(results[0]) == 768
            assert len(results[1]) == 768

            # Test truncate
            with pytest.raises(ValueError):
                results = model.run(texts=["A test sentence." * 10000], truncate=False)

            # Test return_generator=True
            results = model.run(
                texts=["A test sentence.", "Another test sentence."],
                return_generator=True,
            )
            assert isinstance(results, GeneratorType)
            assert len(list(results)) == 2

            # Test unload model
            assert "client" in model.__dict__ and "tokenizer" in model.__dict__
            model.unload_model()
            assert "client" not in model.__dict__ and "tokenizer" not in model.__dict__
