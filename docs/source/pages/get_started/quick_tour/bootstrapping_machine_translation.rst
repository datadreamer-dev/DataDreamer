Bootstrapping Synthetic Few-Shot Examples
#########################################

One technique to perform a task with no examples is to bootstrap synthetic examples that can eventually be used as examples in
a few-shot prompt. Low-resource machine translation (or translation to languages where little to no training data exists) is a
good motivator for this kind of task.

We demonstrate bootstrapping following two papers (`Patel et al., 2022 <https://arxiv.org/abs/2209.14500>`_, `Han et al., 2021 <https://arxiv.org/abs/2110.05448>`_) to translate from English to Tamil with
no paired training examples. Instead, we generate synthetic examples in 4 rounds of bootstrapping.

.. raw:: html

	See the <a class="reference external" href="https://huggingface.co/datasets/datadreamer-dev/english_to_tamil">resulting translations</a>.

.. code-block:: python

	from datadreamer import DataDreamer
	from datadreamer.llms import OpenAI
	from datadreamer.steps import (
		FewShotPrompt,
		ProcessWithPrompt,
		HFHubDataSource,
		CosineSimilarity,
	)
	from datadreamer.embedders import SentenceTransformersEmbedder

	with DataDreamer("./output"):
		# Load GPT-4
		gpt_4 = OpenAI(model_name="gpt-4")

		# Get English sentences
		english_dataset = HFHubDataSource(
			"Get FLORES-101 English Sentences",
			"gsarti/flores_101",
			config_name="eng",
			split="dev",
		).select_columns(["sentence"])

		# Keep only 400 examples as a quick demo
		english_dataset = english_dataset.take(400)

		# Define how many rounds of bootstrapping
		rounds = 4

		# For each round of bootstrapping
		best_translation_pairs = None
		for r in range(rounds):
			# Use a partial set of the sentences for each round
			sentences_for_round = english_dataset.shard(
				num_shards=rounds, index=r, name=f"Round #{r+1}: Get Sentences"
			)

			# Create synthetic pairs
			if r == 0:
				# On the first round, ask GPT-4 to zero-shot translate the English sentences
				# to Tamil to create synthetic translation pairs
				english_to_tamil = ProcessWithPrompt(
					f"Round #{r+1}: Zero-shot Translate from English To Tamil",
					inputs={"inputs": sentences_for_round.output["sentence"]},
					args={
						"llm": gpt_4,
						"input_label": "Sentence:",
						"instruction": "Translate the sentence to Tamil.",
						"max_new_tokens": 1000,
					},
					outputs={"inputs": "english", "generations": "tamil"},
				).select_columns(["english", "tamil"])
			else:
				# On subsequent rounds, use the best synthetic translation pairs from the previous round
				# as few-shot examples to translate more English sentences to create even better synthetic pairs
				english_to_tamil = FewShotPrompt(
					f"Round #{r+1}: Few-shot Translate from English To Tamil",
					inputs={
						"input_examples": best_translation_pairs.output["english"],
						"output_examples": best_translation_pairs.output["tamil"],
						"inputs": sentences_for_round.output["sentence"],
					},
					args={
						"llm": gpt_4,
						"input_label": "English:",
						"output_label": "Tamil:",
						"instruction": "Translate the sentence to Tamil.",
						"max_new_tokens": 1000,
					},
					outputs={"inputs": "english", "generations": "tamil"},
				).select_columns(["english", "tamil"])

			# Automatically filter the best synthetic translation pairs through cosine similarity
			embedder = SentenceTransformersEmbedder("google/mt5-small")
			best_translation_pairs = (
				CosineSimilarity(
					f"Round #{r+1}: Compute Similarities between the Source and Translated Sentences",
					args={"embedder": embedder, "truncate": True},
					inputs={
						"a": english_to_tamil.output["english"],
						"b": english_to_tamil.output["tamil"],
					},
					outputs={"a": "english", "b": "tamil"},
				)
				.sort(
					["similarities"],
					reverse=True,
					name=f"Round #{r+1}: Rank by Similarities",
				)
				.take(2, name=f"Round #{r+1}: Get Top-2 Translation Pairs")
			)

		# Load the test set of English sentences
		english_test_dataset = HFHubDataSource(
			"Get FLORES-101 English Sentences (Test Set)",
			"gsarti/flores_101",
			config_name="eng",
			split="devtest",
		).select_columns(["sentence"])

		# Finally translate the test set with the final bootstrapped synthetic few-shot examples
		english_test_to_tamil = FewShotPrompt(
			"Few-shot Translate from English To Tamil (Test Set)",
			inputs={
				"input_examples": best_translation_pairs.output["english"],
				"output_examples": best_translation_pairs.output["tamil"],
				"inputs": english_test_dataset.output["sentence"],
			},
			args={
				"llm": gpt_4,
				"input_label": "English:",
				"output_label": "Tamil:",
				"instruction": "Translate the sentence to Tamil.",
				"max_new_tokens": 1000,
			},
			outputs={"inputs": "english", "generations": "tamil"},
		).select_columns(["english", "tamil"])

		# Publish and share the synthetic dataset
		english_test_to_tamil.publish_to_hf_hub(
			"datadreamer-dev/english_to_tamil",
		)
