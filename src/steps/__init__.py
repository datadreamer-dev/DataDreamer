"""
:py:class:`Step` objects perform core operations within a DataDreamer session
transforming input data into output data. They are run within a DataDreamer session and
each step takes in data as a set of ``inputs`` and configuration options as a set of
``args`` and then returns output data under the :py:attr:`~Step.output` attribute. A
help message that describes what ``inputs`` and ``args`` a particular step takes in and
what :py:attr:`~Step.output` it returns can be found by accessing the
:py:attr:`~Step.help` string. All steps derive from the :py:class:`Step` base class.

You can :doc:`create your own steps <pages/advanced_usage/creating_a_new_datadreamer_.../step>` to
implement custom logic or a new technique.

Constructing a :py:class:`Step` object
======================================

Step ``inputs``
---------------

Step ``inputs`` are a set of named inputs that are passed to a step via a key-value
dictionary. The input data a step will operate on are given to a step via ``inputs`` and
each input may be of type :py:class:`~datadreamer.datasets.OutputDatasetColumn` or
:py:class:`~datadreamer.datasets.OutputIterableDatasetColumn`. The input columns
passed to a step are typically output columns created by previous steps within a
DataDreamer session.

Step ``args``
-------------

Step ``args`` are a set of named arguments that are passed to a step via a key-value
dictionary. Arguments are non-input data parameters typically used to configure a step's
behavior. Arguments are typically of type ``bool``, ``int``, ``str``, or other Python
types. Some steps like :py:class:`Prompt` or :py:class:`Embed` take
:py:class:`~datadreamer.llms.LLM` objects :py:class:`~datadreamer.embedders.Embedder`
objects as ``args``.


Step ``outputs``
----------------

Steps produce output data in the form of an output dataset object like
:py:class:`~datadreamer.datasets.OutputDataset` or
:py:class:`~datadreamer.datasets.OutputIterableDataset` accessible by the
:py:attr:`~Step.output` attribute. To access a column on these output dataset objects
you can use the ``__getitem__`` operator like so: ``step.output['column_name']``.
This will return a :py:class:`~datadreamer.datasets.OutputDatasetColumn`
or :py:class:`~datadreamer.datasets.OutputIterableDatasetColumn` column object.

Renaming output columns
^^^^^^^^^^^^^^^^^^^^^^^

If you want to rename the columns in the output dataset, at :py:class:`Step`
construction you can pass a ``outputs`` argument that is a dictionary mapping from
the original output column names to the new column names.

Caching
=======

Steps are cached and their outputs are
saved to disk after running. Upon resuming a DataDreamer session, previously completed
steps are loaded from disk.

Types of Steps
==============

DataDreamer comes with a set of built-in steps that can be used to create a workflow
to process data, create synthetic datasets, augment existing datasets, and train models.
We catalogue diferent types of steps below.

.. dropdown:: Data Source

    Most steps take input data and produce output data. Data Source steps are special
    in that they are an initial source of data that can be consumed as input by other
    steps. This could be loading data from an online source, a file, or simply from an
    in-memory ``list`` or ``dict``.

    **Step Classes:**

    - :py:class:`DataSource`
    - :py:class:`HFHubDataSource`
    - :py:class:`HFDatasetDataSource`
    - :py:class:`JSONDataSource`
    - :py:class:`CSVDataSource`
    - :py:class:`TextDataSource`

.. dropdown:: Prompting
    
    There a few different types of prompting steps available. These steps all typically
    take in a :py:class:`~datadreamer.llms.LLM` object as an argument in ``args``.

    .. raw:: html
  
      <h3>Standard Prompting</h3>

    Standard prompting steps help you run prompts against an
    :py:class:`~datadreamer.llms.LLM`.

    **Step Classes:**

    - :py:class:`Prompt`
    - :py:class:`RAGPrompt`
    - :py:class:`ProcessWithPrompt`
    - :py:class:`FewShotPrompt`
    - :py:class:`FewShotPromptWithRetrieval`

    .. raw:: html
  
      <h3>Data Generation Using Prompting</h3>
  
    Some prompting steps are used to help generate completely synthetic data from
    prompts using an instruction.
      
    **Step Classes:**

    - :py:class:`DataFromPrompt`
    - :py:class:`DataFromAttributedPrompt`
      
    .. raw:: html
  
      <h3>Validation & Judging with Prompting</h3>
      
    Some prompting steps are used to help perform validation and judging tasks on a
    set of inputs using an instruction with a :py:class:`~datadreamer.llms.LLM`.
    
    **Step Classes:**
    
    - :py:class:`FilterWithPrompt`
    - :py:class:`RankWithPrompt`
    - :py:class:`JudgePairsWithPrompt`
    - :py:class:`JudgeGenerationPairsWithPrompt`

.. dropdown:: Other NLP Tasks
    
    Miscellaneous steps that help perform other NLP tasks. They typically take in a
    model or engine (like an :py:class:`~datadreamer.embedders.Embedder` object or
    :py:class:`~datadreamer.retrievers.Retriever` object) as an argument in ``args``.

    **Step Classes:**
    
    - :py:class:`Embed`
    - :py:class:`CosineSimilarity`
    - :py:class:`Retrieve`
    - :py:class:`RunTaskModel`
    
.. dropdown:: Training

    See the :doc:`trainers <./datadreamer.trainers>` page for more
    information on training within a DataDreamer session.

Lazy Steps
==========

Some steps may run lazily. This means that the step's output will be lazily
computed as it is consumed by another step. This is useful for steps that work with
large datasets or require expensive computation. Lazy steps will return a
:py:class:`~datadreamer.datasets.OutputIterableDataset` under their
:py:attr:`~Step.output` attribute. At any point, you can use the
:py:attr:`~Step.progress` attribute to get how much of a lazy step's output has been
computed so far.

Lazy steps do not save their output to disk. If you want to explictly save a lazy step's
ouput to disk, you can call :py:meth:`~Step.save` on the lazy step. If you have a series
of lazy steps chained together, you can call :py:meth:`~Step.save` on the last lazy
step to save the final transformed data to disk after all transformations have been
computed, skipping saving intermediate transformations to disk.

Convenience Methods
===================

Various convenience methods are available on :py:class:`Step` objects to help with
frequent tasks.

Previewing a Step's :py:attr:`~Step.output` 
-------------------------------------------
If you want to preview a step's :py:attr:`~Step.output`, you can call the
:py:meth:`~Step.head` method to quickly retrieve a few rows from the step's output
dataset in the form of a :py:class:`pandas.DataFrame` object.

Common Operations
-------------------------------------------
A variety of common data operations and data transformations are available as methods on
:py:class:`Step` objects:

.. list-table::
    :header-rows: 1


    * - Basic Operations
      - Column Operations
      - Sequence Operations
      - Save/Copy Operations
      - Training Dataset Operations
    * - - :py:meth:`~Step.select`
        - :py:meth:`~Step.take`
        - :py:meth:`~Step.skip`
        - :py:meth:`~Step.add_item`
      - - :py:meth:`~Step.select_columns`
        - :py:meth:`~Step.rename_column`
        - :py:meth:`~Step.rename_columns`
        - :py:meth:`~Step.remove_columns` 
      - - :py:meth:`~Step.shuffle`
        - :py:meth:`~Step.sort`
        - :py:meth:`~Step.map`
        - :py:meth:`~Step.filter`
        - :py:meth:`~Step.reverse`
      - - :py:meth:`~Step.save`
        - :py:meth:`~Step.copy`
        - :py:meth:`~Step.shard`
      - - :py:meth:`~Step.splits`

In addition, there are higher-order functions that operate on multiple :py:class:`Step`
objects that can be used to concatenate the output datasets of multiple steps together
or to combine the output dataset columns of multiple steps together (similar to
``zip()`` in Python):

- :py:func:`concat`
- :py:func:`zipped`
    



Running Steps in Parallel
=========================

See the
`Running Steps in Parallel <pages/advanced_usage/parallelization/running_steps_in_parallel.html>`_
page.


Data Card Generation
====================
An automatically generated data card can be viewed by calling
:py:meth:`~Step.data_card`. The data card can be helpful for reproducibility and for
sharing your work with others when published alongside your code. When
`publishing datasets <#exporting-and-publishing-datasets>`_, the data card will be
published alongside the dataset.


The data card traces what steps were run to produce the step's output dataset, what
models were used, what paper citations and software licenses may apply, among other
useful information (see :py:class:`DataCardType`). Reproducibility
information such as the versions of packages used and a fingerprint hash (a signature of
all steps chained together to produce the final step's output dataset) is also included.

Exporting and Publishing Datasets
=================================
You can export the output dataset produced by a step to disk by calling one of the
various export methods available:

- :py:meth:`~Step.export_to_dict()`
- :py:meth:`~Step.export_to_list()`
- :py:meth:`~Step.export_to_json()`
- :py:meth:`~Step.export_to_csv()`
- :py:meth:`~Step.export_to_hf_dataset()`

You can publish the output dataset produced by a step to the
`Hugging Face Hub <https://huggingface.co/>`_ by calling
:py:meth:`~Step.publish_to_hf_hub()`.
"""

from .data_card import DataCardType
from .data_sources.csv_data_source import CSVDataSource
from .data_sources.data_source import DataSource
from .data_sources.hf_dataset_data_source import HFDatasetDataSource
from .data_sources.hf_hub_data_source import HFHubDataSource
from .data_sources.json_data_source import JSONDataSource
from .data_sources.text_data_source import TextDataSource
from .prompt.data_from_attributed_prompt import DataFromAttributedPrompt
from .prompt.data_from_prompt import DataFromPrompt
from .prompt.few_shot_prompt import FewShotPrompt
from .prompt.few_shot_prompt_with_retrieval import FewShotPromptWithRetrieval
from .prompt.filter_with_prompt import FilterWithPrompt
from .prompt.judge_generation_pairs_with_prompt import JudgeGenerationPairsWithPrompt
from .prompt.judge_pairs_with_prompt import JudgePairsWithPrompt
from .prompt.process_with_prompt import ProcessWithPrompt
from .prompt.prompt import Prompt
from .prompt.rag_prompt import RAGPrompt
from .prompt.rank_with_prompt import RankWithPrompt
from .step import LazyRowBatches, LazyRows, Step, SuperStep, concat, zipped
from .step_background import concurrent, wait
from .tasks.cosine_similarity import CosineSimilarity
from .tasks.embed import Embed
from .tasks.retrieve import Retrieve
from .tasks.run_task_model import RunTaskModel

__all__ = [
    "Step",
    "DataSource",
    "HFHubDataSource",
    "HFDatasetDataSource",
    "JSONDataSource",
    "CSVDataSource",
    "TextDataSource",
    "Prompt",
    "RAGPrompt",
    "ProcessWithPrompt",
    "FewShotPrompt",
    "FewShotPromptWithRetrieval",
    "DataFromPrompt",
    "DataFromAttributedPrompt",
    "FilterWithPrompt",
    "RankWithPrompt",
    "JudgePairsWithPrompt",
    "JudgeGenerationPairsWithPrompt",
    "RunTaskModel",
    "Embed",
    "CosineSimilarity",
    "Retrieve",
    "concat",
    "zipped",
    "wait",
    "concurrent",
    "DataCardType",
    "LazyRows",
    "LazyRowBatches",
    "SuperStep",
]
