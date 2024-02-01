Distilling GPT-4 Capabilities to GPT-3.5
########################################

If you want to make GPT-3.5 (a cheaper, smaller, and faster model) more capable, you can use DataDreamer to distill the capabilities of GPT-4 into GPT-3.5. This will allow you to create a more capable model that is cheaper and faster than GPT-4.

We demonstrate an example below on the `ELI5 ("Explain it like I'm 5") <https://www.dictionary.com/e/slang/eli5/>`_ task.

.. code-block:: python

	from datadreamer import DataDreamer
	from datadreamer.llms import OpenAI
	from datadreamer.steps import ProcessWithPrompt, HFHubDataSource
	from datadreamer.trainers import TrainOpenAIFineTune

	with DataDreamer("./output"):
		# Load GPT-4
		gpt_4 = OpenAI(model_name="gpt-4")

		# Get ELI5 questions
		eli5_dataset = HFHubDataSource(
			"Get ELI5 Questions",
			"eli5_category",
			split="train",
			trust_remote_code=True,
		).select_columns(["title"])

		# Keep only 1000 examples as a quick demo
		eli5_dataset = eli5_dataset.take(1000)

		# Ask GPT-4 to ELI5
		questions_and_answers = ProcessWithPrompt(
			"Generate Explanations",
			inputs={"inputs": eli5_dataset.output["title"]},
			args={
				"llm": gpt_4,
				"instruction": (
					'Given the question, give an "Explain it like I\'m 5" answer.'
				),
				"top_p": 1.0,
			},
			outputs={"inputs": "questions", "generations": "answers"},
		)

		# Create training data splits
		splits = questions_and_answers.splits(train_size=0.90, validation_size=0.10)

		# Train a model to answer questions in ELI5 style
		trainer = TrainOpenAIFineTune(
			"Distill capabilities to GPT-3.5",
			model_name="gpt-3.5-turbo-1106",
		)
		trainer.train(
			train_input=splits["train"].output["questions"],
			train_output=splits["train"].output["answers"],
			validation_input=splits["validation"].output["questions"],
			validation_output=splits["validation"].output["answers"],
			epochs=30,
			batch_size=8,
		)

