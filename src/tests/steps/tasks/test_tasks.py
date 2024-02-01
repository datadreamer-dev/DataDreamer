from typing import cast

import pytest
import torch

from ....embedders import SentenceTransformersEmbedder
from ....retrievers import EmbeddingRetriever
from ....steps import (
    CosineSimilarity,
    DataCardType,
    DataSource,
    Embed,
    Retrieve,
    RunTaskModel,
)
from ....task_models import HFClassificationTaskModel


class TestRunTaskModel:
    def test_run_task_model(self, create_datadreamer):
        with create_datadreamer():
            texts = ["A test sentence.", "Another test sentence."]
            model = HFClassificationTaskModel(
                "bdotloh/distilbert-base-uncased-empathetic-dialogues-context"
            )
            dataset = DataSource("Dataset", data={"texts": texts})
            classifications = RunTaskModel(
                "Classifier",
                args={"model": model},
                inputs={"texts": dataset.output["texts"]},
            )
            assert list(classifications.output["texts"]) == texts
            assert "afraid" in list(classifications.output["results"])[0]
            assert classifications._data_card["Classifier"][
                DataCardType.MODEL_NAME
            ] == [model.model_name]
            assert classifications._data_card["Classifier"][
                DataCardType.MODEL_CARD
            ] == [model.model_card]
            assert (
                classifications._data_card["Classifier"][DataCardType.CITATION]
                == model.citation
            )


class TestEmbed:
    def test_embed(self, create_datadreamer):
        with create_datadreamer():
            texts = ["A test sentence.", "Another test sentence."]
            embedder = SentenceTransformersEmbedder(
                "bdotloh/distilbert-base-uncased-empathetic-dialogues-context"
            )
            dataset = DataSource("Dataset", data={"texts": texts})
            embeddings = Embed(
                "Embedder",
                args={"embedder": embedder, "instruction": "Test: "},
                inputs={"texts": dataset.output["texts"]},
            )
            assert list(embeddings.output["texts"]) == texts
            assert len(list(embeddings.output["embeddings"])[0]) == 768
            assert embeddings._data_card["Embedder"][DataCardType.MODEL_NAME] == [
                embedder.model_name
            ]
            assert embeddings._data_card["Embedder"][DataCardType.MODEL_CARD] == [
                embedder.model_card
            ]
            assert (
                embeddings._data_card["Embedder"][DataCardType.CITATION]
                == embedder.citation
            )


class TestCosineSimilarity:
    def test_cosine_sim_between_texts(self, create_datadreamer):
        with create_datadreamer():
            a = ["cat", "cat", "cat", "cat"]
            b = ["cat", "kitten", "dog", "chair"]
            embedder = SentenceTransformersEmbedder(
                "bdotloh/distilbert-base-uncased-empathetic-dialogues-context"
            )
            dataset = DataSource("Dataset", data={"a": a, "b": b})
            a_embeddings = Embed(
                "A Embedder",
                args={"embedder": embedder},
                inputs={"texts": dataset.output["a"]},
            )
            b_embeddings = Embed(
                "B Embedder",
                args={"embedder": embedder},
                inputs={"texts": dataset.output["b"]},
            )
            similarities = CosineSimilarity(
                "Compute cosine similarities",
                inputs={
                    "a": a_embeddings.output["embeddings"],
                    "b": b_embeddings.output["embeddings"],
                },
            )
            assert list(similarities.output["a"]) == list(
                a_embeddings.output["embeddings"]
            )
            assert list(similarities.output["b"]) == list(
                b_embeddings.output["embeddings"]
            )
            assert list(similarities.output["similarities"]) == [
                pytest.approx(1.0000001192092896, 0.0001),
                pytest.approx(0.8328614234924316, 0.0001),
                pytest.approx(0.7612417340278625, 0.0001),
                pytest.approx(0.3415866792201996, 0.0001),
            ]

    def test_cosine_sim_between_embeddings(self, create_datadreamer):
        with create_datadreamer():
            a = ["cat", "cat", "cat", "cat"]
            b = ["cat", "kitten", "dog", "chair"]
            embedder = SentenceTransformersEmbedder(
                "bdotloh/distilbert-base-uncased-empathetic-dialogues-context"
            )
            dataset = DataSource("Dataset", data={"a": a, "b": b})
            similarities = CosineSimilarity(
                "Compute cosine similarities",
                args={"embedder": embedder},
                inputs={"a": dataset.output["a"], "b": dataset.output["b"]},
            )
            assert list(similarities.output["a"]) == a
            assert list(similarities.output["b"]) == b
            assert list(similarities.output["similarities"]) == [
                pytest.approx(1.0000001192092896, 0.0001),
                pytest.approx(0.8328614234924316, 0.0001),
                pytest.approx(0.7612417340278625, 0.0001),
                pytest.approx(0.3415866792201996, 0.0001),
            ]

    def test_cosine_sim_on_multiple_devices(self, create_datadreamer):
        with create_datadreamer():
            a = ["cat", "cat", "cat", "cat"]
            b = ["cat", "kitten", "dog", "chair"]
            embedder = SentenceTransformersEmbedder(
                "bdotloh/distilbert-base-uncased-empathetic-dialogues-context"
            )
            dataset = DataSource("Dataset", data={"a": a, "b": b})
            a_embeddings = Embed(
                "A Embedder",
                args={"embedder": embedder},
                inputs={"texts": dataset.output["a"]},
            )
            b_embeddings = Embed(
                "B Embedder",
                args={"embedder": embedder},
                inputs={"texts": dataset.output["b"]},
            )
            similarities = CosineSimilarity(
                "Compute cosine similarities",
                inputs={
                    "a": a_embeddings.output["embeddings"],
                    "b": b_embeddings.output["embeddings"],
                },
                args={"device": [torch.device("cpu"), torch.device("cpu")]},
            )
            assert list(similarities.output["a"]) == list(
                a_embeddings.output["embeddings"]
            )
            assert list(similarities.output["b"]) == list(
                b_embeddings.output["embeddings"]
            )
            assert list(similarities.output["similarities"]) == [
                pytest.approx(1.0000001192092896, 0.0001),
                pytest.approx(0.8328614234924316, 0.0001),
                pytest.approx(0.7612417340278625, 0.0001),
                pytest.approx(0.3415866792201996, 0.0001),
            ]


class TestRetrieve:
    def test_embed(self, create_datadreamer):
        with create_datadreamer():
            documents = [
                "Dogs bark loudly.",
                "Cats have fur.",
                "Steve Jobs founded Apple.",
            ]
            queries = ["Kittens have fur.", "Bill Gates founded Microsoft."]
            documents_dataset = DataSource("Documents", data={"documents": documents})
            queries_dataset = DataSource("Queries", data={"queries": queries})
            retriever = EmbeddingRetriever(
                texts=documents_dataset.output["documents"],
                embedder=SentenceTransformersEmbedder("all-mpnet-base-v2"),
            )
            results = Retrieve(
                "Retrieve Results",
                args={"retriever": retriever, "k": 2},
                inputs={"queries": queries_dataset.output["queries"]},
            )
            assert list(results.output["queries"]) == queries
            assert len(list(results.output["results"])[0]["indices"]) == 2
            assert results._data_card["Retrieve Results"][DataCardType.MODEL_NAME] == [
                retriever.embedder.model_name
            ]
            assert results._data_card["Retrieve Results"][DataCardType.MODEL_CARD] == [
                retriever.embedder.model_card
            ]
            assert results._data_card["Retrieve Results"][DataCardType.LICENSE] == [
                retriever.license,
                retriever.embedder.license,
            ]
            assert results._data_card["Retrieve Results"][
                DataCardType.CITATION
            ] == cast(list[str], retriever.citation) + cast(
                list[str], retriever.embedder.citation
            )
