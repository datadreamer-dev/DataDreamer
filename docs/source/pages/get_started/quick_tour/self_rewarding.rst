Training a Self-Improving LLM with Self-Rewarding
##################################################

A notable research direction is to use LLMs to self-improve themselves by letting them judge their own outputs and training them to improve. This is similar to :doc:`RLHF <aligning>` except it is `RLAIF (Reinforcement Learning with AI Feedback) <https://arxiv.org/abs/2309.00267>`_. These types of workflows become simple to implement with DataDreamer.

We demonstrate a simple example of the `Self-Rewarding Language Models <https://arxiv.org/abs/2401.10020>`_ paper below.

.. code-block:: python

	from datadreamer import DataDreamer
	from datadreamer.steps import HFHubDataSource, Prompt, JudgeGenerationPairsWithPrompt
	from datadreamer.trainers import TrainHFDPO
	from datadreamer.llms import HFTransformers
	from peft import LoraConfig

	with DataDreamer("./output"):
		# Get a dataset of prompts
		prompts_dataset = HFHubDataSource(
			"Get Prompts Dataset", "Intel/orca_dpo_pairs", split="train"
		).select_columns(["question"])

		# Keep only 3000 examples as a quick demo
		prompts_dataset = prompts_dataset.take(3000)

		# Define how many rounds of self-reward training
		rounds = 3

		# For each round of self-reward training
		adapter_to_apply = None
		for r in range(rounds):
			# Use a partial set of the prompts for each round
			prompts_for_round = prompts_dataset.shard(
				num_shards=rounds, index=r, name=f"Round #{r+1}: Get Prompts"
			)

			# Load the LLM
			llm = HFTransformers(
				"TinyLlama/TinyLlama-1.1B-Chat-v1.0",
				adapter_name=adapter_to_apply,
				device_map="auto",
				dtype="bfloat16",
			)

			# Sample 2 candidate responses from the LLM
			candidate_responses = []
			for candidate_idx in range(2):
				candidate_responses.append(
					Prompt(
						f"Round #{r+1}: Sample Candidate Response #{candidate_idx}",
						inputs={"prompts": prompts_for_round.output["question"]},
						args={
							"llm": llm,
							"batch_size": 2,
							"top_p": 1.0,
							"seed": candidate_idx,
						},
					)
				)

			# Have the LLM judge its own responses
			judgements = JudgeGenerationPairsWithPrompt(
				f"Round #{r+1}: Judge Candidate Responses",
				args={
					"llm": llm,
					"batch_size": 1,
					"max_new_tokens": 5,
				},
				inputs={
					"prompts": prompts_for_round.output["question"],
					"a": candidate_responses[0].output["generations"],
					"b": candidate_responses[1].output["generations"],
				},
			)

			# Unload the LLM
			llm.unload_model()

			# Process the judgements into a preference dataset
			dpo_dataset = judgements.map(
				lambda row: {
					"question": row["prompts"],
					"chosen": row["a"] if row["judgements"] == "Response A" else row["b"],
					"rejected": row["b"] if row["judgements"] == "Response A" else row["a"],
				},
				lazy=False,
				name=f"Round #{r+1}: Create Self-Reward Preference Dataset",
			)

			# Create training data splits
			splits = dpo_dataset.splits(train_size=0.90, validation_size=0.10)

			# Align the TinyLlama chat model with its own preferences
			trainer = TrainHFDPO(
				f"Round #{r+1}: Self-Reward Align TinyLlama-Chat",
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

			# Unload the trained model from memory
			trainer.unload_model()

			# Use the newly trained adapter for the next round of self-reward
			adapter_to_apply = trainer.model_path
