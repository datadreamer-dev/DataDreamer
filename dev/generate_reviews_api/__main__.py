from datadreamer import DataDreamer
from datadreamer.llms import OpenAI
from datadreamer.steps import (
    Prompt,
    DataSource,
    DataFromAttributedPrompt,
)

with DataDreamer("./output"):
    # Load GPT-4
    gpt_4 = OpenAI(model_name="gpt-4")

    # Create prompts to generate attributes for movie reviews
    attribute_generation_prompts = DataSource(
        "Attribute Generation Prompts",
        data={
            "prompts": [
                "過去に劇場公開された3本の映画名をカンマ区切りのリストにしてください。",
                "批評家が考慮すべき映画の要素を3個、カンマ区切りのリストにしてください。",
                "映画批評家のスタイルを表す形容詞をカンマ区切りで3個のリストにしてください。",
            ],
        },
    )

    # Generate the attributes for movie reviews
    attributes = Prompt(
        "Generate Attributes",
        inputs={
            "prompts": attribute_generation_prompts.output["prompts"],
        },
        args={
            "llm": gpt_4,
        },
    ).output["generations"]

    # Generate movie reviews with varied attributes
    movie_reviews = (
        DataFromAttributedPrompt(
            "Generate Movie Reviews",
            args={
                "llm": gpt_4,
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

    # Publish and share the synthetic dataset
    movie_reviews.publish_to_hf_hub(
        "ryota39/movie_reviews",
    )
