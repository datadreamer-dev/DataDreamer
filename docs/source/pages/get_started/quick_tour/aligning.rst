Aligning a LLM with Human Preferences
#####################################

In order to better align the responses :doc:`instruction-tuned LLMs <instruction_tuning>` generate to what humans would prefer, we can train LLMs against a reward model or a dataset of human preferences in a process known as `RLHF (Reinforcement Learning with Human Feedback) <https://arxiv.org/abs/2203.02155>`_.

DataDreamer makes this process extremely simple and straightforward to accomplish. We demonstrate it below using LoRA to only train
a fraction of the weights with `DPO <https://arxiv.org/abs/2305.18290>`_ (a more stable, and efficient alignment method than traditional RLHF).

.. code-block:: python

	from datadreamer import DataDreamer
	from datadreamer.steps import HFHubDataSource
	from datadreamer.trainers import TrainHFDPO
	from peft import LoraConfig

	with DataDreamer("./output"):
		# Get the DPO dataset
		dpo_dataset = HFHubDataSource(
			"Get DPO Dataset", "Intel/orca_dpo_pairs", split="train"
		)

		# Keep only 1000 examples as a quick demo
		dpo_dataset = dpo_dataset.take(1000)

		# Create training data splits
		splits = dpo_dataset.splits(train_size=0.90, validation_size=0.10)

		# Align the TinyLlama chat model with human preferences
		trainer = TrainHFDPO(
			"Align TinyLlama-Chat",
			model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
			peft_config=LoraConfig(),
			device=["cuda:0", "cuda:1"],
			dtype="bfloat16",
		)
		trainer.train(
			train_prompts=splits["train"].output["question"],
			train_chosen=splits["train"].output["chosen"],
			train_rejected=splits["train"].output["rejected"],
			validation_prompts=splits["validation"].output["question"],
			validation_chosen=splits["validation"].output["chosen"],
			validation_rejected=splits["validation"].output["rejected"],
			epochs=3,
			batch_size=1,
			gradient_accumulation_steps=32,
		)
