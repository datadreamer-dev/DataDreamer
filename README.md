<p align="center">
  <a href="#"><img src="/docs/source/_static/images/logo.svg" alt="DataDreamer" style="max-width: 100%;"></a>
</p>

<p align="center">
    <a href="https://circleci.com/gh/huggingface/transformers">
        <img alt="Build" src="https://img.shields.io/circleci/build/github/huggingface/transformers/main">
    </a>
    <a href="https://github.com/huggingface/transformers/blob/main/LICENSE">
        <img alt="GitHub" src="https://img.shields.io/github/license/huggingface/transformers.svg?color=blue">
    </a>
    <a href="https://huggingface.co/docs/transformers/index">
        <img alt="Documentation" src="https://img.shields.io/website/http/huggingface.co/docs/transformers/index.svg?down_color=red&down_message=offline&up_message=online">
    </a>
    <a href="https://github.com/huggingface/transformers/releases">
        <img alt="GitHub release" src="https://img.shields.io/github/release/huggingface/transformers.svg">
    </a>
    <a href="https://github.com/huggingface/transformers/blob/main/CODE_OF_CONDUCT.md">
        <img alt="Contributor Covenant" src="https://img.shields.io/badge/Contributor%20Covenant-v2.0%20adopted-ff69b4.svg">
    </a>
    <a href="https://zenodo.org/badge/latestdoi/155220641"><img src="https://zenodo.org/badge/155220641.svg" alt="DOI"></a>
</p>

<h3 align="center">
    <p>Generate Datasets using Large Language Models</p>
</h3>


<h4>Augment Existing Datasets</h4>

```python
from datadreamer import DataDreamer
from datadreamer.llms import OpenAI
from datadreamer.steps import ZeroShot

with DataDreamer('./'):
    llm = OpenAI("gpt-3.5-turbo")
    trivia_dataset = HFHubDataSource(
        "trivia_dataset", "trivia_qa", "rc", "train"
    ).select_columns(["question", "answer])
    trivia_dataset_processed = trivia_dataset.map(row => {"qa": f"The answer to '{row['question']}' is {row['answer']}."})
    data_with_explanations = ZeroShot(llm, "Explain why the answer is correct.", inputs=trivia_dataset_processed)
    data_with_explanations_verified = data_with_explanations.annotate(num_annotators=3, percent=0.10, active_learning=True)
    data_with_explanations_verified.publish_to_hf_hub("trivia_qa_with_explanations", train_size=1.0)
    # Returns: https://huggingface.co/datasets/trivia_qa_with_explanations
```
