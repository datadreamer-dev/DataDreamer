import torch
from datadreamer import DataDreamer
from datadreamer.llms import HFTransformers
from datadreamer.steps import (
    Prompt,
    DataSource,
    DataFromAttributedPrompt,
    ProcessWithPrompt,
)


session_review = "./output_review"
review_model_name = "sbintuitions/sarashina2.2-1b-instruct-v0.1"
attribute_prompts = [
    "過去に劇場公開された3本の日本の映画名をカンマ区切りのリストにしてください。",
    "批評家が考慮すべき映画の要素を3個、カンマ区切りのリストにしてください。",
    "映画批評家のスタイルを表す形容詞をカンマ区切りで3個のリストにしてください。",
    ]
attribute_instruction = "{movie_name}について、{movie_element}に焦点を当てた{review_style}の映画レビューを数文書きなさい。"
num_sample = 50
review_hf_dataset = "ryota39/movie_reviews_local"


summary_model_name = "sbintuitions/sarashina2.2-1b-instruct-v0.1"
instruction_prompt = "与えられた映画レビューを要約してください。\n\n映画レビュー: {reviews}\n\n要約: "
max_new_tokens = 200


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
            "Generate Movie Reviews",
            args={
                "llm": hf_model,
                "n": num_sample,
                "instruction": attribute_instruction,
                "attributes": {
                    "movie_name": attributes[0].split(","),
                    "movie_element": attributes[1].split(","),
                    "review_style": attributes[2].split(","),
                },
            },
            outputs={"generations": "reviews"},
        )
        .select_columns(["reviews"])
        .shuffle()
    )

    response_model = HFTransformers(
        model_name=summary_model_name,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    reviews_and_summaries = ProcessWithPrompt(
        "Generate Review Responses",
        inputs={
            "inputs": movie_reviews.output["reviews"],
        },
        args={
            "llm": response_model,
            "instruction": instruction_prompt,
            "max_new_tokens": max_new_tokens,
        },
        outputs={"inputs": "reviews", "generations": "summaries"},
    ).select_columns(["reviews", "summaries"])

    reviews_and_summaries.publish_to_hf_hub(
        "ryota39/reviews_and_summaries2",
    )
