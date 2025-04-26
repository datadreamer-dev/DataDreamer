import os
import torch
from datadreamer import DataDreamer
from datadreamer.llms import HFTransformers
from datadreamer.steps import (
    Prompt,
    DataSource,
    DataFromAttributedPrompt,
    ProcessWithPrompt,
)
from dotenv import load_dotenv


load_dotenv()
token = os.environ.get("HF_TOKEN")
session_review = "./output_drill"
review_model_name = "sbintuitions/sarashina2.2-3b-instruct-v0.1"
attribute_prompts = [
    "小学生、中学生、大学生、をカンマ区切りのリストで出力してください。",
    "算数において重要な単語を5単語、カンマ区切りのリストで出力してください。",
    "文章問題、計算問題、選択式問題、をカンマ区切りのリストで出力してください。",
    ]
attribute_instruction = "{term}について、{student_level}に沿った{question_pattern}の問題を書きなさい。"
num_sample = 75

summary_model_name = "sbintuitions/sarashina2.2-1b-instruct-v0.1"
instruction_prompt = "与えられた問題をステップバイステップで答えなさい。\n\n問題: {questions}\n\n回答: "
max_new_tokens = 200
hf_dataset_name = f"ryota39/math-drill"

with DataDreamer(session_review):
    hf_model = HFTransformers(
        model_name=review_model_name,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    attribute_generation_prompts = DataSource(
        "Attribute Generation Prompts",
        data={"prompts": attribute_prompts}
    )

    attributes = Prompt(
        "Generate Attributes",
        inputs={"prompts": attribute_generation_prompts.output["prompts"]},
        args={"llm": hf_model},
    ).output["generations"]

    movie_reviews = (
        DataFromAttributedPrompt(
            "Generate Seed Tasks",
            args={
                "llm": hf_model,
                "n": num_sample,
                "instruction": attribute_instruction,
                "attributes": {
                    "student_level": attributes[0].split(","),
                    "term": attributes[1].split(","),
                    "question_pattern": attributes[2].split(","),
                },
            },
            outputs={"generations": "questions"},
        )
        .select_columns(["questions"])
        .shuffle()
    )

    response_model = HFTransformers(
        model_name=summary_model_name,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    reviews_and_summaries = ProcessWithPrompt(
        "Generate Drill Answer",
        inputs={
            "inputs": movie_reviews.output["questions"],
        },
        args={
            "llm": response_model,
            "instruction": instruction_prompt,
            "max_new_tokens": max_new_tokens,
        },
        outputs={"inputs": "questions", "generations": "answers"},
    ).select_columns(["questions", "answers"])

    reviews_and_summaries.publish_to_hf_hub(
        repo_id=hf_dataset_name,
        private=True,
        token=token,
    )
