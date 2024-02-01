Running Models on Multiple GPUs
#######################################################

There are various ways to run models on multiple GPUs in DataDreamer.

Large LLMs on Multiple GPUs
===========================

To split a large model that cannot fit on a single GPU you can set the ``device_map`` parameter of the
:py:class:`~datadreamer.llms.HFTransformers` class to ``'auto'``. This will automatically split the model by layer
onto your available GPUs. You can also manually specify
`how and where the model should be split <https://huggingface.co/docs/transformers/main/en/main_classes/model#large-model-loading>`_.

Smaller Models
==============

For smaller models, the :py:class:`~datadreamer.llms.ParallelLLM` wrapper takes in multiple :py:class:`~datadreamer.llms.LLM` objects
and behaves like a single unified :py:class:`~datadreamer.llms.LLM` object that can then be passed to a step like :py:class:`~datadreamer.steps.Prompt`. 
:py:class:`~datadreamer.llms.ParallelLLM` will run any inputs it recieves against all of the models in parallel. This is useful for running smaller models on multiple GPUs
as each :py:class:`~datadreamer.llms.LLM` passed to the wrapper can be on a different GPU. Your model must be able to fit on a single GPU
for this to work.

Similarly, we have other parallelization wrappers for other types of models like :py:class:`~datadreamer.embedders.ParallelEmbedder`,
:py:class:`~datadreamer.retrievers.ParallelRetriever`, etc.
