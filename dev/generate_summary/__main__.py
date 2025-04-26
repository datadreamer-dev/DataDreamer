import torch
from datadreamer import DataDreamer
from datadreamer.llms import HFTransformers
from datadreamer.steps import ProcessWithPrompt, HFHubDataSource


with DataDreamer("./output"):
    movie_reviews = HFHubDataSource(
        "映画レビューコーパスのロード",
        path="ryota39/movie_reviews_local",
        split="train",
    ).select_columns(["reviews"])

    response_model = HFTransformers(
        model_name="sbintuitions/sarashina2.2-1b-instruct-v0.1",
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    instruction_prompt = "与えられた映画レビューを要約してください。\n\n映画レビュー: {reviews}\n\n要約: "

    reviews_and_summaries = ProcessWithPrompt(
        "Generate Review Responses",
        inputs={
            "inputs": movie_reviews.output["reviews"],
        },
        args={
            "llm": response_model,
            "instruction": instruction_prompt,
            "max_new_tokens": 200,
        },
        outputs={"inputs": "reviews", "generations": "summaries"},
    ).select_columns(["reviews", "summaries"])

    reviews_and_summaries.publish_to_hf_hub(
        "ryota39/reviews_and_summaries",
    )
