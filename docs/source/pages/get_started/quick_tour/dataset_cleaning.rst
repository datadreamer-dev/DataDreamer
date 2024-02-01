Cleaning an Existing Dataset
############################

DataDreamer can help clean or filter existing datasets using LLMs. We demonstrate this below by filtering a dataset of
news articles to only include those that are about sports.

.. raw:: html

	See the resulting <a class="reference external" href="https://huggingface.co/datasets/datadreamer-dev/cnn_dailymail_sports">synthetic dataset</a>.

.. code-block:: python
    
	from datadreamer import DataDreamer
	from datadreamer.llms import OpenAI
	from datadreamer.steps import FilterWithPrompt, HFHubDataSource

	with DataDreamer("./output"):
		# Load GPT-4
		gpt_4 = OpenAI(model_name="gpt-4")

		# Get news articles
		news_dataset = HFHubDataSource(
			"Get CNN & Daily Mail News Articles",
			"cnn_dailymail",
			config_name="3.0.0",
			split="test",
		)

		# Keep only 1000 articles as a quick demo
		news_dataset = news_dataset.take(1000)

		# Ask GPT-4 to filter the dataset
		sports_news_dataset = FilterWithPrompt(
			"Filter to only keep sports articles",
			inputs={"inputs": news_dataset.output["article"]},
			args={
				"llm": gpt_4,
				"instruction": "Is the article about sports? Answer 'Yes' or 'No'.",
			},
		)

		# Publish and share the synthetic dataset
		sports_news_dataset.publish_to_hf_hub(
			"datadreamer-dev/cnn_dailymail_sports",
		)
