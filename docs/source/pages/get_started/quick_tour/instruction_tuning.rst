Instruction-Tuning a LLM
########################

When LLMs are pre-trained, they are pre-trained in a self-supervised mannner to simply predict the next word in a sentence. This can yield
a model that can follow human instructions to some degree, but is often not very effective until this "base model" is fined-tuned on a dataset
of example instructions and responses in a process known as `"instruction-tuning" <https://arxiv.org/pdf/2203.02155.pdf>`_, essentially allowing the model to learn to follow natural
language instructions.

DataDreamer makes this process extremely simple and straightforward to accomplish. We demonstrate it below using LoRA to only train
a fraction of the weights.

.. code-block:: python

	from datadreamer import DataDreamer
	from datadreamer.steps import HFHubDataSource
	from datadreamer.trainers import TrainHFFineTune
	from peft import LoraConfig

	with DataDreamer("./output"):
		# Get the Alpaca instruction-tuning dataset (cleaned version)
		instruction_tuning_dataset = HFHubDataSource(
			"Get Alpaca Instruction-Tuning Dataset", "yahma/alpaca-cleaned", split="train"
		)

		# Keep only 1000 examples as a quick demo
		instruction_tuning_dataset = instruction_tuning_dataset.take(1000)

		# Some examples taken in an "input", we'll format those into the instruction
		instruction_tuning_dataset.map(
			lambda row: {
				"instruction": (
					row["instruction"]
					if len(row["input"]) == 0
					else f"Input: {row['input']}\n\n{row['instruction']}"
				),
				"output": row["output"],
			},
			lazy=False,
		)

		# Create training data splits
		splits = instruction_tuning_dataset.splits(train_size=0.90, validation_size=0.10)

		# Define what the prompt template should be when instruction-tuning
		chat_prompt_template = "### Instruction:\n{{prompt}}\n\n### Response:\n"

		# Instruction-tune the base TinyLlama model to make it follow instructions
		trainer = TrainHFFineTune(
			"Instruction-Tune TinyLlama",
			model_name="TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
			chat_prompt_template=chat_prompt_template,
			peft_config=LoraConfig(),
			device=["cuda:0", "cuda:1"],
			dtype="bfloat16",
		)
		trainer.train(
			train_input=splits["train"].output["instruction"],
			train_output=splits["train"].output["output"],
			validation_input=splits["validation"].output["instruction"],
			validation_output=splits["validation"].output["output"],
			epochs=3,
			batch_size=1,
			gradient_accumulation_steps=32,
		)
