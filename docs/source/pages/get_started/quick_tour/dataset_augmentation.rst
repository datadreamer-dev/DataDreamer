Augmenting an Existing Dataset
##############################

DataDreamer can help augment existing datasets using LLMs. We demonstrate this below by augmenting questions from HotpotQA
with a decomposition of what steps a user would need to take to solve the complex question.

.. raw:: html

	See the resulting <a class="reference external" href="https://huggingface.co/datasets/datadreamer-dev/hotpot_qa_augmented">synthetic dataset</a>.

.. code-block:: python
    
	from datadreamer import DataDreamer
	from datadreamer.llms import OpenAI
	from datadreamer.steps import ProcessWithPrompt, HFHubDataSource

	with DataDreamer("./output"):
		# Load GPT-4
		gpt_4 = OpenAI(model_name="gpt-4")

		# Get HotPot QA questions
		hotpot_qa_dataset = HFHubDataSource(
			"Get Hotpot QA Questions",
			"hotpot_qa",
			config_name="distractor",
			split="train",
		).select_columns(["question"])

		# Keep only 1000 questions as a quick demo
		hotpot_qa_dataset = hotpot_qa_dataset.take(1000)

		# Ask GPT-4 to decompose the question
		questions_and_decompositions = ProcessWithPrompt(
			"Generate Decompositions",
			inputs={"inputs": hotpot_qa_dataset.output["question"]},
			args={
				"llm": gpt_4,
				"instruction": (
					"Given the question which requires multiple steps to solve, give a numbered list of intermediate questions required to solve the question."
					"Return only the list, nothing else."
				),
			},
			outputs={"inputs": "questions", "generations": "decompositions"},
		).select_columns(["questions", "decompositions"])

		# Publish and share the synthetic dataset
		questions_and_decompositions.publish_to_hf_hub(
			"datadreamer-dev/hotpot_qa_augmented",
		)
