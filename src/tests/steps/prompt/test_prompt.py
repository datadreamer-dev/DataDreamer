import sys
from itertools import islice

import pytest

from ...._cachable._cachable import _StrWithSeed
from ....datasets import OutputIterableDataset
from ....embedders import SentenceTransformersEmbedder
from ....llms import HFTransformers, OpenAI
from ....retrievers import EmbeddingRetriever
from ....steps import (
    DataCardType,
    DataFromAttributedPrompt,
    DataFromPrompt,
    DataSource,
    FewShotPrompt,
    FewShotPromptWithRetrieval,
    FilterWithPrompt,
    JudgeGenerationPairsWithPrompt,
    JudgePairsWithPrompt,
    ProcessWithPrompt,
    Prompt,
    RAGPrompt,
    RankWithPrompt,
    wait,
)


class TestPromptBase:
    def test_prompt_base(self, create_datadreamer):
        # This test seems to throw a warning from Python on macOS / M-series chips:
        #
        # https://github.com/apple/ml-stable-diffusion/issues/8
        #
        # It looks like an active bug with M-series chips / macOS / CPython.
        # There is nothing we can do to fix it, the following warning can be ignored:
        # """
        #  multiprocessing/resource_tracker.py:224: UserWarning: resource_tracker:
        #  There appear to be 1 leaked semaphore objects to clean up at shutdown
        #  warnings.warn('resource_tracker: There appear to be
        # """
        with create_datadreamer():
            prompts = ["What color is the sky?", "Who was the first president"]
            llm = HFTransformers("google/flan-t5-small")
            questions = DataSource("Questions Dataset", data={"questions": prompts})
            answers = Prompt(
                "Answer Questions",
                args={"llm": llm, "max_new_tokens": 2, "batch_size": 5},
                inputs={"prompts": questions.output["questions"]},
                outputs={"generations": "answers"},
                background=True,
            )
            wait(answers)
            assert questions.fingerprint == "ec0e94db113f0bb0"
            assert answers.fingerprint == "1d0d52db3487c081"
            assert list(answers.output["prompts"]) == prompts
            assert list(answers.output["answers"]) == ["blue", "john"]
            assert answers._data_card["Answer Questions"][DataCardType.MODEL_NAME] == [
                llm.model_name
            ]
            assert answers._data_card["Answer Questions"][DataCardType.MODEL_CARD] == [
                llm.model_card
            ]
            assert answers._data_card["Answer Questions"][DataCardType.LICENSE] == [
                llm.license
            ]
            assert (
                answers._data_card["Answer Questions"][DataCardType.CITATION]
                == llm.citation
            )

    def test_prompt_base_lazy(self, create_datadreamer, mock_llm):
        with create_datadreamer():
            prompts = {
                "What color is the sky?": "blue",
                "Who was the first president?": "George Washington",
            }
            llm = mock_llm(HFTransformers("google/flan-t5-small"), prompts)
            questions = DataSource(
                "Questions Dataset",
                data={
                    "questions": list(prompts.keys())
                    + ["Extra Prompt That Will Throw Error If Reached"]
                },
            )
            answers = Prompt(
                "Answer Questions",
                args={
                    "llm": llm,
                    "batch_size": 2,
                    "batch_scheduler_buffer_size": 2,
                    "lazy": True,
                },
                inputs={"prompts": questions.output["questions"]},
                outputs={"generations": "answers"},
            )
            assert isinstance(answers.output, OutputIterableDataset)
            assert list(islice(iter(answers.output["prompts"]), 2)) == list(
                prompts.keys()
            )
            assert list(islice(iter(answers.output["answers"]), 2)) == list(
                prompts.values()
            )

    def test_prompt_logging(self, create_datadreamer, mock_llm, caplog):
        with create_datadreamer():
            prompts = {"0": "0", "1": "1", "2": "2"}
            llm = mock_llm(HFTransformers("google/flan-t5-small"), prompts)
            questions = DataSource(
                "Questions Dataset", data={"questions": [str(i) for i in range(3)]}
            )
            Prompt(
                "Answer Questions",
                args={"llm": llm, "post_process": lambda p: p.upper(), "batch_size": 1},
                inputs={"prompts": questions.output["questions"]},
                outputs={"prompts": "questions", "generations": "answers"},
                progress_interval=0,
            )
            logs = [rec.message for rec in caplog.records]
            caplog.clear()
            assert len(logs) == 8
            assert any(
                ["Step 'Answer Questions' progress: 66% 🔄" in log for log in logs]
            )

    def test_prompt_logging_iterable(self, create_datadreamer, mock_llm, caplog):
        with create_datadreamer():
            prompts = {"0": "0", "1": "1", "2": "2"}
            llm = mock_llm(HFTransformers("google/flan-t5-small"), prompts)

            def dataset_generator():
                for i in range(3):
                    yield {"questions": str(i)}

            questions = DataSource(
                "Questions Dataset", data=dataset_generator, auto_progress=False
            )
            Prompt(
                "Answer Questions",
                args={"llm": llm, "post_process": lambda p: p.upper(), "batch_size": 1},
                inputs={"prompts": questions.output["questions"]},
                outputs={"prompts": "questions", "generations": "answers"},
                progress_interval=0,
            )
            logs = [rec.message for rec in caplog.records]
            caplog.clear()
            assert len(logs) == 9
            assert "Step 'Answer Questions' progress: 2 row(s) 🔄" in logs


class TestPrompt:
    def test_prompt(self, create_datadreamer, mock_llm):
        with create_datadreamer():
            prompts = {
                "What color is the sky?": "blue",
                "Who was the first president?": "George Washington",
            }
            llm = mock_llm(OpenAI("gpt-3.5-turbo-instruct"), prompts)
            questions = DataSource(
                "Questions Dataset", data={"questions": list(prompts.keys())}
            )
            answers = Prompt(
                "Answer Questions",
                args={"llm": llm, "post_process": lambda p: p.upper()},
                inputs={"prompts": questions.output["questions"]},
                outputs={"prompts": "questions", "generations": "answers"},
            )
            assert answers.output.column_names == ["questions", "answers"]
            assert list(answers.output["questions"]) == list(prompts.keys())
            assert list(answers.output["answers"]) == list(
                map(lambda p: p.upper(), prompts.values())
            )


class TestRAGPrompt:
    def test_rag_prompt(self, create_datadreamer, mock_llm):
        with create_datadreamer():
            prompts = {
                "Here are some documents.\n\nDocument: The sun is yellow.\nDocument: The moon is white.\n\nQuestion: What color is the sun?": "yellow",
                "Here are some documents.\n\nDocument: Steve Jobs founded Apple.\nDocument: Bill Gates founded Microsoft.\n\nQuestion: Who founded Apple?": "Steve Jobs",
            }
            llm = mock_llm(OpenAI("gpt-3.5-turbo-instruct"), prompts)
            documents_dataset = DataSource(
                "Documents",
                data={
                    "documents": [
                        "The sun is yellow.",
                        "The moon is white.",
                        "Steve Jobs founded Apple.",
                        "Bill Gates founded Microsoft.",
                    ]
                },
            )
            retriever = EmbeddingRetriever(
                texts=documents_dataset.output["documents"],
                embedder=SentenceTransformersEmbedder("all-mpnet-base-v2"),
            )
            questions = DataSource(
                "Questions Dataset",
                data={"questions": ["What color is the sun?", "Who founded Apple?"]},
            )
            answers = RAGPrompt(
                "Answer Questions",
                args={
                    "llm": llm,
                    "retriever": retriever,
                    "k": 2,
                    "beg_instruction": "Here are some documents.",
                },
                inputs={"prompts": questions.output["questions"]},
                outputs={"prompts": "questions", "generations": "answers"},
            )
            assert answers.output.column_names == ["questions", "answers"]
            assert list(answers.output["questions"]) == list(prompts.keys())
            assert list(answers.output["answers"]) == list(prompts.values())


class TestDataFromPrompt:
    def test_data_from_prompt(self, create_datadreamer, mock_llm):
        with create_datadreamer():
            prompts = {
                _StrWithSeed("Generate a sentence.", seed=0): "First sentence.",
                _StrWithSeed("Generate a sentence.", seed=1): "Second sentence.",
                _StrWithSeed("Generate a sentence.", seed=2): "Third sentence.",
                _StrWithSeed("Generate a sentence.", seed=3): "Fourth sentence.",
            }
            llm = mock_llm(OpenAI("gpt-3.5-turbo-instruct"), prompts)
            answers = DataFromPrompt(
                "Generate sentences",
                args={
                    "llm": llm,
                    "n": 4,
                    "instruction": "Generate a sentence.",
                    "batch_size": 2,
                },
                outputs={"generations": "sentences"},
            )
            assert answers.output.column_names == ["prompts", "sentences"]
            assert list(answers.output["prompts"]) == [
                str(p) for p in list(prompts.keys())
            ]
            assert list(answers.output["sentences"]) == [
                "First sentence.",
                "Second sentence.",
                "Third sentence.",
                "Fourth sentence.",
            ]

    @pytest.mark.skipif(
        sys.platform == "darwin", reason="instable on macOS/M2 (floating point diffs)"
    )
    def test_data_from_prompt_with_seed(self, create_datadreamer, mocker):
        with create_datadreamer():
            prompts = {
                _StrWithSeed("Generate a sentence.", seed=0): "As you can see,",
                _StrWithSeed("Generate a sentence.", seed=1): "You may be prompted to",
                _StrWithSeed("Generate a sentence.", seed=2): "Examples:",
                _StrWithSeed("Generate a sentence.", seed=3): "1. Def",
            }
            llm = HFTransformers("gpt2")
            answers = DataFromPrompt(
                "Generate sentences",
                args={
                    "llm": llm,
                    "n": 4,
                    "instruction": "Generate a sentence.",
                    "batch_size": 2,
                    "max_new_tokens": 5,
                    "seed": 43,
                },
                outputs={"generations": "sentences"},
            )
            assert answers.output.column_names == ["prompts", "sentences"]
            assert list(answers.output["prompts"]) == [
                str(p) for p in list(prompts.keys())
            ]
            assert list(answers.output["sentences"]) == list(prompts.values())


class TestDataFromAttributedPrompt:
    def test_data_from_attributed_prompt(self, create_datadreamer, mock_llm):
        with create_datadreamer():
            prompts = {
                _StrWithSeed(
                    "Generate a sentence that is serious and short.", seed=0
                ): "First sentence.",
                _StrWithSeed(
                    "Generate a sentence that is serious and long.", seed=0
                ): "Second sentence.",
                _StrWithSeed(
                    "Generate a sentence that is funny and short.", seed=0
                ): "Third sentence.",
                _StrWithSeed(
                    "Generate a sentence that is funny and long.", seed=0
                ): "Fourth sentence.",
            }
            llm = mock_llm(OpenAI("gpt-3.5-turbo-instruct"), prompts)
            answers = DataFromAttributedPrompt(
                "Generate sentences",
                args={
                    "llm": llm,
                    "n": 4,
                    "instruction": "Generate a sentence that is {adjective} and {length}.",
                    "attributes": {
                        "adjective": ["serious", "funny"],
                        "length": ["short", "long"],
                    },
                    "batch_size": 2,
                },
                outputs={"generations": "sentences"},
            )
            assert answers.output.column_names == ["attributes", "prompts", "sentences"]
            assert list(answers.output["attributes"]) == [
                {"adjective": "serious", "length": "short"},
                {"adjective": "serious", "length": "long"},
                {"adjective": "funny", "length": "short"},
                {"adjective": "funny", "length": "long"},
            ]
            assert list(answers.output["prompts"]) == [
                str(p) for p in list(prompts.keys())
            ]
            assert list(answers.output["sentences"]) == [
                "First sentence.",
                "Second sentence.",
                "Third sentence.",
                "Fourth sentence.",
            ]

    def test_data_from_attributed_prompt_with_list_of_combinations(
        self, create_datadreamer, mock_llm
    ):
        with create_datadreamer():
            prompts = {
                _StrWithSeed(
                    "Generate a sentence that is serious and short.", seed=0
                ): "First sentence.",
                _StrWithSeed(
                    "Generate a sentence that is serious and long.", seed=0
                ): "Second sentence.",
                _StrWithSeed(
                    "Generate a sentence that is funny and short.", seed=0
                ): "Third sentence.",
                _StrWithSeed(
                    "Generate a sentence that is funny and long.", seed=0
                ): "Fourth sentence.",
            }
            llm = mock_llm(OpenAI("gpt-3.5-turbo-instruct"), prompts)
            answers = DataFromAttributedPrompt(
                "Generate sentences",
                args={
                    "llm": llm,
                    "n": 4,
                    "instruction": "Generate a sentence that is {adjective} and {length}.",
                    "attributes": [
                        {"adjective": "serious", "length": "short"},
                        {"adjective": "serious", "length": "long"},
                        {"adjective": "funny", "length": "short"},
                        {"adjective": "funny", "length": "long"},
                    ],
                    "batch_size": 2,
                },
                outputs={"generations": "sentences"},
            )
            assert answers.output.column_names == ["attributes", "prompts", "sentences"]
            assert list(answers.output["attributes"]) == [
                {"adjective": "serious", "length": "short"},
                {"adjective": "serious", "length": "long"},
                {"adjective": "funny", "length": "short"},
                {"adjective": "funny", "length": "long"},
            ]
            assert list(answers.output["prompts"]) == [
                str(p) for p in list(prompts.keys())
            ]
            assert list(answers.output["sentences"]) == [
                "First sentence.",
                "Second sentence.",
                "Third sentence.",
                "Fourth sentence.",
            ]


class TestProcessWithPrompt:
    def test_process_with_prompt(self, create_datadreamer, mock_llm):
        with create_datadreamer():
            prompts = {
                "Input: 3\n\nInstruction: Add one to the input.": "4",
                "Input: 4\n\nInstruction: Add one to the input.": "5",
            }
            llm = mock_llm(OpenAI("gpt-3.5-turbo-instruct"), prompts)
            inputs = DataSource("To Predict", data={"x": list(range(3, 5))})
            answers = ProcessWithPrompt(
                "Add one",
                args={"llm": llm, "instruction": "Add one to the input."},
                inputs={"inputs": inputs.output["x"]},
                outputs={"generations": "x+1"},
            )
            assert answers.output.column_names == ["inputs", "prompts", "x+1"]
            assert list(answers.output["inputs"]) == list(inputs.output["x"])
            assert list(answers.output["prompts"]) == list(prompts.keys())
            assert list(answers.output["x+1"]) == list(prompts.values())


class TestFilterWithPrompt:
    def test_filter_with_prompt(self, create_datadreamer, mock_llm):
        with create_datadreamer():
            prompts = {
                "Input: 1\n\nInstruction: Is the number greater than 1?": "No!",
                "Input: 2\n\nInstruction: Is the number greater than 1?": "Yes!",
            }
            llm = mock_llm(OpenAI("gpt-3.5-turbo-instruct"), prompts)
            inputs = DataSource(
                "To Filter", data={"x": list(range(1, 3)), "y": ["a", "b"]}
            )
            filtered = FilterWithPrompt(
                "Filter Greater Than",
                args={"llm": llm, "instruction": "Is the number greater than 1?"},
                inputs={"inputs": inputs.output["x"]},
                outputs={"y": "z"},
            )
            assert filtered.output.column_names == ["x", "z"]
            assert list(filtered.output["x"]) == list(inputs.output["x"])[1:]
            assert list(filtered.output["z"]) == list(inputs.output["y"])[1:]


class TestRankWithPrompt:
    def test_rank_with_prompt(self, create_datadreamer, mock_llm):
        with create_datadreamer():
            prompts = {
                "Input: chair\n\nInstruction: How funny is this word on a scale of 1-10?": "Not funny, 0",
                "Input: octothorpe\n\nInstruction: How funny is this word on a scale of 1-10?": "Medium: 5",
                "Input: hullaballoo\n\nInstruction: How funny is this word on a scale of 1-10?": "I would say, 10!",
            }
            llm = mock_llm(OpenAI("gpt-3.5-turbo-instruct"), prompts)
            inputs = DataSource(
                "To Filter",
                data={
                    "x": ["chair", "octothorpe", "hullaballoo"],
                    "y": ["a", "b", "c"],
                },
            )
            filtered = RankWithPrompt(
                "Rank Inputs",
                args={
                    "llm": llm,
                    "instruction": "How funny is this word on a scale of 1-10?",
                    "score_threshold": 4,
                },
                inputs={"inputs": inputs.output["x"]},
                outputs={"y": "z", "scores": "funny_scores"},
            )
            assert filtered.output.column_names == ["x", "z", "funny_scores"]
            assert list(filtered.output["x"]) == ["hullaballoo", "octothorpe"]
            assert list(filtered.output["z"]) == ["c", "b"]
            assert list(filtered.output["funny_scores"]) == [10, 5]


class TestJudgePairsWithPrompt:
    def test_judge_pairs_with_prompt(self, create_datadreamer, mock_llm):
        with create_datadreamer():
            prompts = {
                "Input A: A good response.\n\nInput B: A bad response.\n\nInstruction: Which input is better? Respond with 'Input A' or 'Input B'.": "Probably, Input A",
                "Input A: A worse response.\n\nInput B: A great response.\n\nInstruction: Which input is better? Respond with 'Input A' or 'Input B'.": "I don't know",
                "Input A: A terrible response.\n\nInput B: A wonderful response.\n\nInstruction: Which input is better? Respond with 'Input A' or 'Input B'.": "Input B",
            }
            llm = mock_llm(OpenAI("gpt-3.5-turbo-instruct"), prompts)
            inputs = DataSource(
                "To Judge",
                data={
                    "a": [
                        "A good response.",
                        "A worse response.",
                        "A wonderful response.",
                    ],
                    "b": [
                        "A bad response.",
                        "A great response.",
                        "A terrible response.",
                    ],
                },
            )
            answers = JudgePairsWithPrompt(
                "Judge Pairs",
                args={"llm": llm},
                inputs={"a": inputs.output["a"], "b": inputs.output["b"]},
                outputs={"judgements": "choosen"},
            )
            assert answers.output.column_names == [
                "a",
                "b",
                "judge_prompts",
                "judge_generations",
                "choosen",
            ]
            assert list(answers.output["a"]) == list(inputs.output["a"])
            assert list(answers.output["b"]) == list(inputs.output["b"])
            assert list(answers.output["judge_prompts"]) == list(prompts.keys())
            assert list(answers.output["judge_generations"]) == list(prompts.values())
            assert list(answers.output["choosen"]) == ["Input A", "Input B", "Input A"]


class TestJudgeGenerationPairsWithPrompt:
    def test_judge_generation_pairs_with_prompt(self, create_datadreamer, mock_llm):
        with create_datadreamer():
            prompts = {
                "Prompt: Do something.\n\nResponse A: A good response.\n\nResponse B: A bad response.\n\nInstruction: Which response is better? Respond with 'Response A' or 'Response B'.": "Probably, Response A",
                "Prompt: Do something else.\n\nResponse A: A worse response.\n\nResponse B: A great response.\n\nInstruction: Which response is better? Respond with 'Response A' or 'Response B'.": "I don't know",
                "Prompt: Do something again.\n\nResponse A: A terrible response.\n\nResponse B: A wonderful response.\n\nInstruction: Which response is better? Respond with 'Response A' or 'Response B'.": "Response B",
            }
            llm = mock_llm(OpenAI("gpt-3.5-turbo-instruct"), prompts)
            inputs = DataSource(
                "To Judge",
                data={
                    "prompts": [
                        "Do something.",
                        "Do something else.",
                        "Do something again.",
                    ],
                    "a": [
                        "A good response.",
                        "A worse response.",
                        "A wonderful response.",
                    ],
                    "b": [
                        "A bad response.",
                        "A great response.",
                        "A terrible response.",
                    ],
                },
            )
            answers = JudgeGenerationPairsWithPrompt(
                "Judge Generation Pairs",
                args={"llm": llm},
                inputs={
                    "prompts": inputs.output["prompts"],
                    "a": inputs.output["a"],
                    "b": inputs.output["b"],
                },
                outputs={"judgements": "choosen"},
            )
            assert answers.output.column_names == [
                "prompts",
                "a",
                "b",
                "judge_prompts",
                "judge_generations",
                "choosen",
            ]
            assert list(answers.output["prompts"]) == list(inputs.output["prompts"])
            assert list(answers.output["a"]) == list(inputs.output["a"])
            assert list(answers.output["b"]) == list(inputs.output["b"])
            assert list(answers.output["judge_prompts"]) == list(prompts.keys())
            assert list(answers.output["judge_generations"]) == list(prompts.values())
            assert list(answers.output["choosen"]) == [
                "Response A",
                "Response B",
                "Response A",
            ]


class TestFewShotPrompt:
    def test_few_shot_prompt(self, create_datadreamer, mock_llm):
        with create_datadreamer():
            prompts = {
                "Input: 0\nOutput: 0\nInput: 1\nOutput: 2\nInput: 2\nOutput: 4\nInput: 3\nOutput:": "6",  # noqa: B950
                "Input: 0\nOutput: 0\nInput: 1\nOutput: 2\nInput: 2\nOutput: 4\nInput: 4\nOutput:": "8",  # noqa: B950
            }
            llm = mock_llm(OpenAI("gpt-3.5-turbo-instruct"), prompts)
            examples = DataSource(
                "Examples Dataset",
                data={"x": list(range(3)), "y": list(map(lambda x: 2 * x, range(3)))},
            )
            inputs = DataSource("To Predict", data={"x": list(range(3, 5))})
            answers = FewShotPrompt(
                "Answer Questions",
                args={"llm": llm},
                inputs={
                    "input_examples": examples.output["x"],
                    "output_examples": examples.output["y"],
                    "inputs": inputs.output["x"],
                },
                outputs={"generations": "y_predict"},
            )
            assert answers.output.column_names == ["inputs", "prompts", "y_predict"]
            assert list(answers.output["inputs"]) == list(inputs.output["x"])
            assert list(answers.output["prompts"]) == list(prompts.keys())
            assert list(answers.output["y_predict"]) == list(prompts.values())

    def test_few_shot_prompt_custom(self, create_datadreamer, mock_llm):
        with create_datadreamer():
            prompts = {
                "Double the number.\n\nX: 0\nY: 0\nX: 1\nY: 2\nX: 3\nY:": "6",
                "Double the number.\n\nX: 0\nY: 0\nX: 1\nY: 2\nX: 4\nY:": "8",
            }
            llm = mock_llm(OpenAI("gpt-3.5-turbo-instruct"), prompts)
            examples = DataSource(
                "Examples Dataset",
                data={"x": list(range(3)), "y": list(map(lambda x: 2 * x, range(3)))},
            )
            inputs = DataSource("To Predict", data={"x": list(range(3, 5))})
            answers = FewShotPrompt(
                "Answer Questions",
                args={
                    "llm": llm,
                    "input_label": "X:",
                    "output_label": "Y:",
                    "min_in_context_examples": 1,
                    "max_in_context_examples": 2,
                    "instruction": "Double the number.",
                },
                inputs={
                    "input_examples": examples.output["x"],
                    "output_examples": examples.output["y"],
                    "inputs": inputs.output["x"],
                },
                outputs={"generations": "y_predict"},
            )
            assert answers.output.column_names == ["inputs", "prompts", "y_predict"]
            assert list(answers.output["inputs"]) == list(inputs.output["x"])
            assert list(answers.output["prompts"]) == list(prompts.keys())
            assert list(answers.output["y_predict"]) == list(prompts.values())


class TestFewShotPromptWithRetrieval:
    def test_few_shot_prompt_with_retrieval(self, create_datadreamer, mock_llm):
        with create_datadreamer():
            prompts = {
                "Input: Great\nOutput: positive\nInput: Good\nOutput: positive\nInput: Terrific\nOutput:": "positive",  # noqa: B950
                "Input: Terrible\nOutput: negative\nInput: Bad\nOutput: negative\nInput: Awful\nOutput:": "negative",  # noqa: B950
            }
            llm = mock_llm(OpenAI("gpt-3.5-turbo-instruct"), prompts)
            embedder = SentenceTransformersEmbedder("all-mpnet-base-v2")
            examples = DataSource(
                "Examples Dataset",
                data={
                    "words": ["Bad", "Good", "Terrible", "Great"],
                    "sentiment": ["negative", "positive", "negative", "positive"],
                },
            )
            inputs = DataSource("To Predict", data={"words": ["Terrific", "Awful"]})
            answers = FewShotPromptWithRetrieval(
                "Answer Questions",
                args={"llm": llm, "embedder": embedder, "k": 2},
                inputs={
                    "input_examples": examples.output["words"],
                    "output_examples": examples.output["sentiment"],
                    "inputs": inputs.output["words"],
                },
                outputs={"generations": "sentiment"},
            )
            assert answers.output.column_names == ["inputs", "prompts", "sentiment"]
            assert list(answers.output["inputs"]) == list(inputs.output["words"])
            assert list(answers.output["prompts"]) == list(prompts.keys())
            assert list(answers.output["sentiment"]) == list(prompts.values())
