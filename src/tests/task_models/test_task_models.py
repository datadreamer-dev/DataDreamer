import os
from types import GeneratorType

import pytest

from ... import DataDreamer
from ...task_models import HFClassificationTaskModel


class TestHFClassificationTaskModel:
    def test_init(self, create_datadreamer):
        with create_datadreamer():
            db_path = os.path.join(
                DataDreamer.get_output_folder_path(),
                ".cache",
                "HFClassificationTaskModel_bdotloh-distilbert-base-uncased-empathetic"
                "-dialogues-context_torch.float32.db",
            )
            model = HFClassificationTaskModel(
                "bdotloh/distilbert-base-uncased-empathetic-dialogues-context"
            )
            cache, _ = model.cache_and_lock  # type: ignore[misc]
            assert os.path.exists(db_path)

    def test_metadata(self, create_datadreamer):
        with create_datadreamer():
            model = HFClassificationTaskModel(
                "bdotloh/distilbert-base-uncased-empathetic-dialogues-context"
            )
            assert model.model_card is not None
            assert model.license is None
            assert isinstance(model.citation, list)
            assert len(model.citation) == 1

    def test_run(self, create_datadreamer):
        with create_datadreamer():
            # Simple test
            model = HFClassificationTaskModel(
                "bdotloh/distilbert-base-uncased-empathetic-dialogues-context"
            )
            assert model.model_max_length == 512
            results = model.run(texts=["I am so happy.", "I am so angry."])
            assert isinstance(results, list)
            assert results[0]["joyful"] > 0.95 and results[0]["angry"] < 0.05
            assert results[1]["angry"] > 0.95 and results[1]["joyful"] < 0.30

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

    def test_peft(self, create_datadreamer):
        with create_datadreamer():
            # Simple test
            model = HFClassificationTaskModel(
                "distilbert-base-uncased",
                adapter_name="samadpls/sentiment-analysis",
                num_labels=2,
                label2id={"negative": 0, "positive": 1},
                id2label={0: "negative", 1: "positive"},
            )
            assert isinstance(model.citation, list)
            assert len(model.citation) == 3
            assert model.model_max_length == 512
            results = model.run(texts=["I love that.", "I hate that."])
            assert isinstance(results, list)
            assert results[0]["positive"] > 0.99 and results[0]["negative"] < 0.01
            assert results[1]["negative"] > 0.99 and results[1]["positive"] < 0.01

            # Test unload model
            assert "model" in model.__dict__ and "tokenizer" in model.__dict__
            model.unload_model()
            assert "model" not in model.__dict__ and "tokenizer" not in model.__dict__
