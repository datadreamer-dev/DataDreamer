Overview Guide
#######################################################

The following is a high-level conceptual overview of the DataDreamer library to help you quickly understand how different
components work together to create prompting, synthetic data generation, and training workflows.

.. seealso::

    **For concrete examples and recipes**: see the :doc:`Quick Tour <../../pages/get_started/quick_tour/index>` page.

    **For a detailed API reference**: see the :doc:`API Reference <../../datadreamer>` page.

DataDreamer Sessions
====================

Any code you write with the DataDreamer library, you will place within a `DataDreamer session <../../datadreamer.html#datadreamer-sessions>`_ like so:

.. code-block:: python

        from datadreamer import DataDreamer

        with DataDreamer('./output/'):
            # ... run steps or trainers here ...

**Within a session, you can run any steps or trainers you want.** DataDreamer will automatically organize, cache, and save the
results of each `step <../../datadreamer.steps.html#caching>`__ or `trainer <../../datadreamer.trainers.html#resuming-training>`__ run within a session to the output folder. This makes the session easily **resumable** if interrupted and
**reproducible** when the `code is shared along with session the output folder <#reproducibility>`_.

Steps
-----

A :doc:`step <../../datadreamer.steps>` in DataDreamer transforms some input data to some output data. **Steps are the core operators in a
DataDreamer session** and are useful for generating data from LLMs, synthetic data augmentation to existing datasets, or any other data
processing task. The output of one step can be used as the input to another step, allowing you to **chain together multiple steps to create
complex workflows**.

For example, the :py:class:`~datadreamer.steps.HFHubDataSource` step lets you load in an existing dataset from the Hugging Face Hub. Steps
like :py:class:`~datadreamer.steps.Prompt`, :py:class:`~datadreamer.steps.FewShotPrompt`, and :py:class:`~datadreamer.steps.FewShotPromptWithRetrieval`
help you produce generations from from LLMs. You can see `all of the available built-in steps here <../../datadreamer.steps.html#types-of-steps>`_. Although not
required to use DataDreamer, you may be interested in :doc:`creating your own steps <../../pages/advanced_usage/creating_a_new_datadreamer_.../step>` to encapsulate a custom technique
or routine.

Trainers
--------

A :doc:`trainer <../../datadreamer.trainers>` in DataDreamer can train on a dataset, usually the output of a step, and produces a model. **Trainers are useful for alignment,
fine-tuning, instruction-tuning, training classifiers, and training a model from scratch.**

Many types of training schemes are supported including for example :py:class:`~datadreamer.trainers.TrainHFFineTune`, :py:class:`~datadreamer.trainers.TrainSentenceTransformer`, :py:class:`~datadreamer.trainers.TrainHFDPO`. You can see the full list of :doc:`available trainers here <../../datadreamer.trainers>`. Trainers also support
`training on multiple GPUs <../../datadreamer.trainers.html#training-on-multiple-gpus>`_ and `training with quantization <../../datadreamer.trainers.html#quantization>`_ and `parameter-efficient techniques like LoRA <../../datadreamer.trainers.html#parameter-efficient-training>`_.

Models
------

You can instantiate models like a :py:class:`~datadreamer.llms.LLM` or :py:class:`~datadreamer.embedders.Embedder`.
A step may require a model as an argument to run, for example, :py:class:`~datadreamer.steps.Prompt` takes a :py:class:`~datadreamer.llms.LLM` as an argument.
**Models make it easy to
load and run the latest open source models as well as models served by API** 
(`OpenAI <https://openai.com/>`_,
`Anthropic <https://www.anthropic.com/>`_,
`Together AI <https://www.together.ai/>`_,
`Mistral AI <https://mistral.ai/>`_,
etc.). DataDreamer also makes it **simple to swap models** for experimentation. You can see the full list of :doc:`available LLMs here <../../datadreamer.llms>`.

DataDreamer provides a variety of utilities around
`efficient generation <../../datadreamer.llms.html#efficient-generation-techniques>`_ including loading them with
`quantization <../../datadreamer.llms.html#quantization>`_,
`running them on multiple GPUs <../../datadreamer.llms.html#running-on-multiple-gpus>`_, 
`caching generations <../../datadreamer.llms.html#caching>`_,
and more.



Publishing Datasets and Models
--------------------------------------------

DataDreamer makes it extremely simple and easy to export and publish both `datasets <../../datadreamer.steps.html#exporting-and-publishing-datasets>`_ (the outputs of steps) and `models <../../datadreamer.trainers.html#exporting-and-publishing-models>`_ (the outputs of trainers) to the
`Hugging Face Hub <https://huggingface.co/>`_. This makes it **easier to openly share datasets and models you create with others** along with reproducibility information.

DataDreamer will also automatically generate and publish `data cards <../../datadreamer.steps.html#data-card-generation>`_ and `model cards <../../datadreamer.trainers.html#model-card-generation>`_ that contain **useful metadata like software license information, citations, and reproducibility information**
alongside your dataset or model.

Reproducibility
===============

DataDreamer has a strong focus on reproducibility and **DataDreamer sessions are easily reproducible** when the `code and the session output folder <#datadreamer-sessions>`_ is shared alongside
any published datasets or models created by DataDreamer.

.. important::

    By sharing the session output folder, it **allows others to reproduce or extend your workflow** by easily resuming
    the DataDreamer session and modifying it, while taking advantage of :doc:`cached intermediate outputs and work <../../pages/advanced_usage/caching_and_saved_outputs>` to avoid expensive
    or slow re-computation where possible.

Datasets and models published with DataDreamer also have automatically generated `data cards <../../datadreamer.steps.html#data-card-generation>`_ and `model cards <../../datadreamer.trainers.html#model-card-generation>`_ for reproducibility.

Advanced Usage
==============

Now that you have a basic understanding of the DataDreamer library, you may be interested topics covered in the :doc:`Advanced Usage <../../pages/advanced_usage/index>` section.

