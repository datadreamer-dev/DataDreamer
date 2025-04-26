import torch
from datadreamer import DataDreamer
from datadreamer.llms import HFTransformers
from datadreamer.steps import (
    Prompt,
    DataSource,
    DataFromAttributedPrompt,
)

with DataDreamer("./output"):
    # Hugging Faceのモデルをロード
    japanese_model = HFTransformers(
        model_name="sbintuitions/sarashina2.2-1b-instruct-v0.1",
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    # 映画レビューの属性を生成するためのプロンプト
    attribute_generation_prompts = DataSource(
        "Attribute Generation Prompts",
        data={
            "prompts": [
                "過去に劇場公開された3本の日本の映画名をカンマ区切りのリストにしてください。",
                "批評家が考慮すべき映画の要素を3個、カンマ区切りのリストにしてください。",
                "映画批評家のスタイルを表す形容詞をカンマ区切りで3個のリストにしてください。",
            ],
        },
    )

    # 映画レビューの属性を生成
    attributes = Prompt(
        "Generate Attributes",
        inputs={
            "prompts": attribute_generation_prompts.output["prompts"],
        },
        args={
            "llm": japanese_model,
        },
    ).output["generations"]

    # 生成された属性を使って映画レビューを生成
    movie_reviews = (
        DataFromAttributedPrompt(
            "Generate Movie Reviews",
            args={
                "llm": japanese_model,
                "n": 50,
                "instruction": "{movie_name}について、{movie_element}に焦点を当てた{review_style}の映画レビューを数文書きなさい。",
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

    movie_reviews.publish_to_hf_hub(
        "ryota39/movie_reviews_local",
    )
