# type: ignore
# flake8: noqa

en_data = HFDataset("en_c4").select({"en_text": "text"}).limit(100)
es_data = HFDataset("es_c4").select({"es_text": "text"}).limit(100)

to_es_zs = ZeroShot(
    args={
        "instruction": "Translate the input text to es."
    }
    inputs={
        "queries": en_data.output["en_text"]
    }
    outputs= {  
        "responses": "es_text"
    } 
)

en_to_es_fs_examples = (
    Dataset(
        {
            "en_text": to_es_zs.input,
            "es_text": to_es_zs.output,
        }
    )
    .shuffle()
    .limit(3)
)

for _ in range(5):
    to_es_fs = FewShot(
        examples=(
            en_to_es_fs_examples.output.en_text,
            en_to_es_fs_examples.output.es_text,
        ),
        inputs=en_data,
        outputs="es_text",
    )

    es_to_en_fs_examples = (
        Dataset(
            {
                "es_text": to_es_fs.output,
                "en_text": to_es_fs.input,
            }
        )
        .shuffle()
        .limit(3)
    )

    to_en_fs = FewShot(
        examples=(
            es_to_en_fs_examples.output.es_text,
            es_to_en_fs_examples.output.en_text,
        ),
        inputs=es_data,
        outputs="en_text",
    )

    en_to_es_fs_examples = (
        Dataset(
            {
                "en_text": to_en_fs.output,
                "es_text": to_en_fs.input,
            }
        )
        .shuffle()
        .limit(3)
    )

to_es_fs = FewShot(
    examples=(en_to_es_fs_examples.output.en_text, en_to_es_fs_examples.output.es_text),
    inputs=en_data,
    outputs="es_text",
)

synthetic_es_to_en_dataset = Dataset(
    {
        "es_text": to_es_fs.output,
        "en_text": to_es_fs.input,
    }
)

# =================================
# Auto-generator

# inputs:
#     en_text: generate({
#         'format':'English sentences.',
#         'content': 'A variety of topics surrounding daily life, politics, entertainment, news, conversation, etc.',
#         'style': 'A variety of styles, from more formal language, to casual language.'
#         'example': 'I went to the park yesterday.',
#         'length': 5,
#     })
# task: "Translate to Spanish" # acts as a Zero-Shot Prompt
# boosting: {
#     "few-shot-amplification": 5,
#     "s": 5,
# }
