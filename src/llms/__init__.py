"""
:py:class:`LLM` objects help run an LLM on prompts. All LLMs derive from the
:py:class:`LLM` base class.

.. tip::

   Instead of using :py:meth:`~LLM.run` directly, use a
   :py:class:`step <datadreamer.steps>` that takes an :py:class:`LLM` as an ``args``
   argument such as :py:class:`~datadreamer.steps.Prompt` or
   :py:class:`~datadreamer.steps.FewShotPrompt`.

Efficient Generation Techniques
===============================

Throughput
----------

DataDreamer provides efficient generation through a variety of techniques that can
optimize throughput.

.. dropdown:: Adaptive Batch Sizing

    For locally running LLMs, the maximum batch size that the LLM can handle before 
    running out of memory is determined by the amount of memory available on your
    system, but also dependent on the maximum sequence length the batch of inputs passed
    to the LLM, as longer inputs require more memory. Over many iterations, DataDreamer
    will automatically learn the maximum batch size that the LLM can handle for a given
    sequence length and will adaptively adjust the batch size to maximize throughput.

    The maximum batch size that will ever be used is the ``batch_size`` argument passed
    to the :py:meth:`~LLM.run` method. DataDreamer will try to find the largest batch
    size that the LLM can handle that is less than or equal to the ``batch_size``. If
    a batch size is too large, DataDreamer will automatically catch the out of memory
    error and reduce the batch size and learn for future iterations.

    To disable adaptive batch sizing, you can pass ``adaptive_batch_size=False`` to the
    :py:meth:`~LLM.run` method.

.. dropdown:: Batch Scheduling

    In order to minimize padding processed by the LLM, DataDreamer will attempt to
    schedule batches such that the length of all sequences in a batch are similar. This
    will minimize the amount of padding that the LLM has to process.

    To do this, DataDreamer reads a large buffer of prompts, sorts the prompts by
    length, and then schedules batches of size ``batch_size`` from the sorted prompts.
    To manually control the size of the buffer, you can pass a
    ``batch_scheduler_buffer_size`` to the :py:meth:`~LLM.run` method.

    To disable batch scheduling, you can set ``batch_scheduler_buffer_size`` equal to
    ``batch_size``.
    
.. dropdown:: Robustness & Retries 

    For API-based LLMs, DataDreamer will attempt to retry failed requests. This is can
    be disabled via ``retry_on_fail=False``.



Running on Multiple GPUs
------------------------
See the
`Running Models on Multiple GPUs <pages/advanced_usage/parallelization/running_models_on_multiple_gpus.html>`_
page.

Quantization
------------
See the
`Quantization <pages/advanced_usage/quantization.html>`_
page.

Caching
=======
LLMs internally perform caching to disk, so if you run the same prompt with the same
generation settings multiple times, the LLM will only run the prompt once and then
cache the results for future runs.
"""

from .ai21 import AI21
from .anthropic import Anthropic
from .bedrock import Bedrock
from .cohere import Cohere
from .ctransformers import CTransformers
from .hf_api_endpoint import HFAPIEndpoint
from .hf_transformers import HFTransformers
from .llm import LLM
from .mistral_ai import MistralAI
from .openai import OpenAI
from .openai_assistant import OpenAIAssistant
from .palm import PaLM
from .parallel_llm import ParallelLLM
from .petals import Petals
from .together import Together
from .vertex_ai import VertexAI
from .vllm import VLLM

__all__ = [
    "LLM",
    "OpenAI",
    "OpenAIAssistant",
    "HFTransformers",
    "CTransformers",
    "VLLM",
    "Petals",
    "HFAPIEndpoint",
    "Together",
    "MistralAI",
    "Anthropic",
    "Cohere",
    "AI21",
    "Bedrock",
    "PaLM",
    "VertexAI",
    "ParallelLLM",
]
