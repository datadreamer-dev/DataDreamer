import os
from types import GeneratorType

import pytest

from ... import DataDreamer
from ...embedders import SentenceTransformersEmbedder
from ...retrievers import EmbeddingRetriever, ParallelRetriever
from ...steps import DataSource


class TestEmbeddingRetriever:
    def test_init(self, create_datadreamer):
        with create_datadreamer():
            db_path = os.path.join(
                DataDreamer.get_output_folder_path(),
                ".cache",
                "EmbeddingRetriever_all-mpnet-base-v2.db",
            )
            documents = DataSource(
                "Documents",
                data={
                    "documents": [
                        "Dogs bark loudly.",
                        "Cats have fur.",
                        "Steve Jobs founded Apple.",
                    ]
                },
            ).output["documents"]
            retriever = EmbeddingRetriever(
                texts=documents,
                embedder=SentenceTransformersEmbedder("all-mpnet-base-v2"),
            )
            cache, _ = retriever.cache_and_lock  # type: ignore[misc]
            assert os.path.exists(db_path)

    def test_metadata(self, create_datadreamer):
        with create_datadreamer():
            documents = DataSource(
                "Documents",
                data={
                    "documents": [
                        "Dogs bark loudly.",
                        "Cats have fur.",
                        "Steve Jobs founded Apple.",
                    ]
                },
            ).output["documents"]
            retriever = EmbeddingRetriever(
                texts=documents,
                embedder=SentenceTransformersEmbedder("all-mpnet-base-v2"),
            )
            assert retriever.model_card is None
            assert retriever.license is not None
            assert isinstance(retriever.citation, list)
            assert len(retriever.citation) == 1

    def test_index(self, create_datadreamer, caplog):
        with create_datadreamer():
            # Simple test
            documents = DataSource(
                "Documents",
                data={
                    "documents": [
                        "Dogs bark loudly.",
                        "Cats have fur.",
                        "Steve Jobs founded Apple.",
                    ]
                },
            ).output["documents"]
            retriever = EmbeddingRetriever(
                texts=documents,
                embedder=SentenceTransformersEmbedder("all-mpnet-base-v2"),
            )
            results = retriever.run(
                queries=["Kittens have fur.", "Bill Gates founded Microsoft."], k=2
            )
            assert results == [
                {
                    "indices": [1, 0],
                    "texts": ["Cats have fur.", "Dogs bark loudly."],
                    "scores": [
                        pytest.approx(0.81670237, 0.0001),
                        pytest.approx(0.1688818, 0.0001),
                    ],
                },
                {
                    "indices": [2, 1],
                    "texts": ["Steve Jobs founded Apple.", "Cats have fur."],
                    "scores": [
                        pytest.approx(0.7169484, 0.0001),
                        pytest.approx(0.1270127, 0.0001),
                    ],
                },
            ]
            logs = [rec.message for rec in caplog.records]
            caplog.clear()
            assert "Building index." in logs
            assert "Writing index to disk." in logs
            assert "Finished writing index to disk." in logs
            retriever_index_path = os.path.join(
                DataDreamer.get_output_folder_path(),
                ".cache",
                "retrievers",
                "EmbeddingRetriever_all-mpnet-base-v2_1.0",
                "5e486236fcdf5dcb",
            )
            assert os.path.isfile(os.path.join(retriever_index_path, "faiss.index"))
            assert os.path.isfile(os.path.join(retriever_index_path, "faiss.db"))

            # Test index is cached
            retriever = EmbeddingRetriever(
                texts=documents,
                embedder=SentenceTransformersEmbedder("all-mpnet-base-v2"),
            )
            results = retriever.run(
                queries=["Bill Gates founded Microsoft.", "Kittens have fur."],
                k=2,
                force=True,
            )
            assert results == [
                {
                    "indices": [2, 1],
                    "texts": ["Steve Jobs founded Apple.", "Cats have fur."],
                    "scores": [
                        pytest.approx(0.7169484, 0.0001),
                        pytest.approx(0.1270127, 0.0001),
                    ],
                },
                {
                    "indices": [1, 0],
                    "texts": ["Cats have fur.", "Dogs bark loudly."],
                    "scores": [
                        pytest.approx(0.81670237, 0.0001),
                        pytest.approx(0.1688818, 0.0001),
                    ],
                },
            ]
            logs = [rec.message for rec in caplog.records]
            caplog.clear()
            assert len(logs) == 0

            # Test unload model
            assert "index" in retriever.__dict__
            retriever.unload_model()
            assert "index" not in retriever.__dict__

    def test_in_memory_index(self, create_datadreamer, caplog):
        with create_datadreamer():
            # Simple test
            documents = DataSource(
                "Documents",
                data={
                    "documents": [
                        "Dogs bark loudly.",
                        "Cats have fur.",
                        "Steve Jobs founded Apple.",
                    ]
                },
            ).output["documents"]

        with create_datadreamer(":memory:"):
            retriever = EmbeddingRetriever(
                texts=documents,
                embedder=SentenceTransformersEmbedder("all-mpnet-base-v2"),
            )
            results = retriever.run(
                queries=["Kittens have fur.", "Bill Gates founded Microsoft."], k=2
            )
            assert results == [
                {
                    "indices": [1, 0],
                    "texts": ["Cats have fur.", "Dogs bark loudly."],
                    "scores": [
                        pytest.approx(0.81670237, 0.0001),
                        pytest.approx(0.1688818, 0.0001),
                    ],
                },
                {
                    "indices": [2, 1],
                    "texts": ["Steve Jobs founded Apple.", "Cats have fur."],
                    "scores": [
                        pytest.approx(0.7169484, 0.0001),
                        pytest.approx(0.1270127, 0.0001),
                    ],
                },
            ]
            logs = [rec.message for rec in caplog.records]
            caplog.clear()
            assert "Building index." in logs
            assert "Writing index to disk." not in logs
            assert "Finished writing index to disk." not in logs

            # Test index is cached
            retriever = EmbeddingRetriever(
                texts=documents,
                embedder=SentenceTransformersEmbedder("all-mpnet-base-v2"),
            )
            results = retriever.run(
                queries=["Bill Gates founded Microsoft.", "Kittens have fur."], k=2
            )
            assert results == [
                {
                    "indices": [2, 1],
                    "texts": ["Steve Jobs founded Apple.", "Cats have fur."],
                    "scores": [
                        pytest.approx(0.7169484, 0.0001),
                        pytest.approx(0.1270127, 0.0001),
                    ],
                },
                {
                    "indices": [1, 0],
                    "texts": ["Cats have fur.", "Dogs bark loudly."],
                    "scores": [
                        pytest.approx(0.81670237, 0.0001),
                        pytest.approx(0.1688818, 0.0001),
                    ],
                },
            ]
            logs = [rec.message for rec in caplog.records]
            caplog.clear()
            assert "Building index." in logs
            assert "Writing index to disk." not in logs
            assert "Finished writing index to disk." not in logs

    def test_run(self, create_datadreamer):
        with create_datadreamer():
            # Simple test with instructions
            documents = DataSource(
                "Documents",
                data={
                    "documents": [
                        "Dogs bark loudly.",
                        "Cats have fur.",
                        "Steve Jobs founded Apple.",
                    ]
                },
            ).output["documents"]
            retriever = EmbeddingRetriever(
                texts=documents,
                embedder=SentenceTransformersEmbedder("all-mpnet-base-v2"),
                index_instruction="Represent the document for indexing: ",
                query_instruction="Represent the document for querying: ",
            )
            results = retriever.run(
                queries=["Kittens have fur.", "Bill Gates founded Microsoft."], k=2
            )
            assert results == [
                {
                    "indices": [1, 0],
                    "texts": ["Cats have fur.", "Dogs bark loudly."],
                    "scores": [
                        pytest.approx(0.84728813, 0.0001),
                        pytest.approx(0.70916176, 0.0001),
                    ],
                },
                {
                    "indices": [2, 0],
                    "texts": ["Steve Jobs founded Apple.", "Dogs bark loudly."],
                    "scores": [
                        pytest.approx(0.8433093, 0.0001),
                        pytest.approx(0.570959, 0.0001),
                    ],
                },
            ]

            # Test truncate
            retriever = EmbeddingRetriever(
                texts=documents,
                embedder=SentenceTransformersEmbedder("all-mpnet-base-v2"),
                truncate=False,
            )
            with pytest.raises(ValueError):
                results = retriever.run(queries=["A test sentence." * 10000])

            # Test return_generator=True
            retriever = EmbeddingRetriever(
                texts=documents,
                embedder=SentenceTransformersEmbedder("all-mpnet-base-v2"),
            )
            results = retriever.run(
                queries=["Kittens have fur.", "Bill Gates founded Microsoft."],
                k=2,
                return_generator=True,
            )
            assert isinstance(results, GeneratorType)
            assert len(list(results)) == 2


class TestParallelRetriever:
    def test_run(self, create_datadreamer):
        with create_datadreamer():
            # Simple test
            documents = DataSource(
                "Documents",
                data={
                    "documents": [
                        "Dogs bark loudly.",
                        "Cats have fur.",
                        "Steve Jobs founded Apple.",
                    ]
                },
            ).output["documents"]
            retriever_1 = EmbeddingRetriever(
                texts=documents,
                embedder=SentenceTransformersEmbedder("all-mpnet-base-v2"),
            )
            retriever_2 = EmbeddingRetriever(
                texts=documents,
                embedder=SentenceTransformersEmbedder("all-mpnet-base-v2"),
            )
            parallel_retriever = ParallelRetriever(retriever_1, retriever_2)
            results = parallel_retriever.run(
                queries=["Kittens have fur.", "Bill Gates founded Microsoft."],
                k=2,
                batch_size=1,
            )
            assert results == [
                {
                    "indices": [1, 0],
                    "texts": ["Cats have fur.", "Dogs bark loudly."],
                    "scores": [
                        pytest.approx(0.81670237, 0.0001),
                        pytest.approx(0.1688818, 0.0001),
                    ],
                },
                {
                    "indices": [2, 1],
                    "texts": ["Steve Jobs founded Apple.", "Cats have fur."],
                    "scores": [
                        pytest.approx(0.7169484, 0.0001),
                        pytest.approx(0.1270127, 0.0001),
                    ],
                },
            ]

            # Test return_generator=True
            results = parallel_retriever.run(
                queries=["Kittens have fur.", "Bill Gates founded Microsoft."],
                k=2,
                batch_size=1,
                return_generator=True,
            )
            assert isinstance(results, GeneratorType)
            assert len(list(results)) == 2

            # Unload model
            parallel_retriever.unload_model()
