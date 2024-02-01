Training an "Abstract to Tweet Model" with Fully Synthetic Data
###############################################################

In this demonstration, we show how to train a small model to generate tweets summarizing ML research paper abstracts. We use DataDreamer to generate a fully synthetic dataset, distill the knowledge to a small T5 model, and publish both the dataset and model.

.. raw:: html

	See the resulting <a class="reference external" href="https://huggingface.co/datasets/datadreamer-dev/abstracts_and_tweets">synthetic dataset</a> and <a class="reference external" href="https://huggingface.co/datadreamer-dev/abstracts_to_tweet_model">the trained model</a>.

.. code-block:: python

	from datadreamer import DataDreamer
	from datadreamer.llms import OpenAI
	from datadreamer.steps import DataFromPrompt, ProcessWithPrompt
	from datadreamer.trainers import TrainHFFineTune
	from peft import LoraConfig

	with DataDreamer("./output"):
		# Load GPT-4
		gpt_4 = OpenAI(model_name="gpt-4")

		# Generate synthetic arXiv-style research paper abstracts with GPT-4
		arxiv_dataset = DataFromPrompt(
			"Generate Research Paper Abstracts",
			args={
				"llm": gpt_4,
				"n": 1000,
				"temperature": 1.2,
				"instruction": (
					"Generate an arXiv abstract of an NLP research paper."
					" Return just the abstract, no titles."
				),
			},
			outputs={"generations": "abstracts"},
		)

		# Ask GPT-4 to convert the abstracts to tweets
		abstracts_and_tweets = ProcessWithPrompt(
			"Generate Tweets from Abstracts",
			inputs={"inputs": arxiv_dataset.output["abstracts"]},
			args={
				"llm": gpt_4,
				"instruction": (
					"Given the abstract, write a tweet to summarize the work."
				),
				"top_p": 1.0,
			},
			outputs={"inputs": "abstracts", "generations": "tweets"},
		)

		# Create training data splits
		splits = abstracts_and_tweets.splits(train_size=0.90, validation_size=0.10)

		# Train a model to convert research paper abstracts to tweets
		# with the synthetic dataset
		trainer = TrainHFFineTune(
			"Train an Abstract => Tweet Model",
			model_name="google/t5-v1_1-base",
			peft_config=LoraConfig(),
		)
		trainer.train(
			train_input=splits["train"].output["abstracts"],
			train_output=splits["train"].output["tweets"],
			validation_input=splits["validation"].output["abstracts"],
			validation_output=splits["validation"].output["tweets"],
			epochs=30,
			batch_size=8,
		)

		# Publish and share the synthetic dataset
		abstracts_and_tweets.publish_to_hf_hub(
			"datadreamer-dev/abstracts_and_tweets",
			train_size=0.90,
			validation_size=0.10,
		)

		# Publish and share the trained model
		trainer.publish_to_hf_hub("datadreamer-dev/abstracts_to_tweet_model")