"""
:py:class:`Trainer` objects help train a model on a dataset within a DataDreamer
session. All trainers derive from the :py:class:`Trainer` base class. Trainers typically
have a :py:meth:`~Trainer.train` method that takes
:py:class:`~datadreamer.datasets.OutputDatasetColumn` or
:py:class:`~datadreamer.datasets.OutputIterableDatasetColumn` dataset columns as
inputs and outputs for training. Dataset columns can be accessed from the
`output produced by steps <./datadreamer.steps.html#step-outputs>`_.

Resuming Training
=================
For trainers that support resumability, trainers will resume training from the last
checkpoint saved automatically if interrupted.

Efficient Training Techniques
=============================

Training on Multiple GPUs
-------------------------
DataDreamer makes training on multiple GPUs with extremely simple and straightforward.
All you need to do is pass in a list of devices to the ``device`` parameter of
:py:class:`Trainer` at construction instead of a single device. That's it.
For more advanced configuration, see the
`Training on Multiple GPUs <pages/advanced_usage/parallelization/training_models_on_multiple_gpus.html>`_
page.

Quantization
------------
See the
`Quantization <pages/advanced_usage/quantization.html>`_
page.

Parameter-Efficient Training 
----------------------------
See the
`Parameter-Efficient Training <pages/advanced_usage/parameter_efficient_training.html>`_
page.

Configuring Training
====================

You can make use of ``**kwargs`` to pass in additional keyword arguments to the
underlying model's training method.

.. dropdown:: Configuring Early Stopping

    To override the default early stopping, pass in the ``early_stopping_patience``
    parameter and the ``early_stopping_threshold`` parameter. To disable early stopping,
    set ``early_stopping_patience`` to ``None``.

    See :py:class:`~transformers.EarlyStoppingCallback` for more details.

.. dropdown:: Advanced Configuration

    Most configuration you need can be done by passing ``**kwargs``. However, if you need
    more advanced configuration, you can subclass the trainer class and override as
    needed or :doc:`create your own trainer
    <pages/advanced_usage/creating_a_new_datadreamer_.../trainer>`.

Model Card Generation
=====================
An automatically generated model card can be viewed by calling
:py:meth:`~Trainer.model_card`. The model card can be helpful for reproducibility and
for sharing your work with others when published alongside your code. When
`publishing models <#exporting-and-publishing-models>`_, the model card will be
published alongside the model.


The model card traces what steps were run to produce the model's training dataset, what
models were used, what paper citations and software licenses may apply, among other
useful information. Reproducibility information such as the versions of packages used
and a fingerprint hash (a signature of all steps chained together to produce the final
training dataset) is also included.

Exporting and Publishing Models
===============================
You can export a trained model to disk by calling ``export_to_disk()``. You can publish
a trained model to the `Hugging Face Hub <https://huggingface.co/>`_ by calling
``publish_to_hf_hub()``. DataDreamer will also helpfully include instructions with a
code snippet on how to load and use your model and setup the `demonstration widget on
the Hugging Face Hub model page <https://huggingface.co/docs/hub/models-widgets>`_ with
some examples.

"""

from .train_hf_classifier import TrainHFClassifier
from .train_hf_dpo import TrainHFDPO
from .train_hf_finetune import TrainHFFineTune
from .train_hf_ppo import TrainHFPPO
from .train_hf_reward_model import TrainHFRewardModel
from .train_openai_finetune import TrainOpenAIFineTune
from .train_sentence_transformer import TrainSentenceTransformer
from .train_setfit_classifier import TrainSetFitClassifier
from .trainer import Trainer

__all__ = [
    "Trainer",
    "TrainOpenAIFineTune",
    "TrainHFClassifier",
    "TrainHFFineTune",
    "TrainSentenceTransformer",
    "TrainHFDPO",
    "TrainHFRewardModel",
    "TrainHFPPO",
    "TrainSetFitClassifier",
]
