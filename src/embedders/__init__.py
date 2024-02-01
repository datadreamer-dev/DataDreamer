"""
:py:class:`Embedder` objects help convert texts to :wikipedia:`embeddings <Word_embedding>`.
All embedders derive from the :py:class:`Embedder` base class.

.. tip::

   Instead of using :py:meth:`~Embedder.run` directly, use a
   :py:class:`step <datadreamer.steps>` that takes an :py:class:`Embedder` as an ``args``
   argument such as :py:class:`~datadreamer.steps.Embed` or construct an
   :py:class:`~datadreamer.retrievers.EmbeddingRetriever` with the embedder and then use
   a retrieval step such as :py:class:`~datadreamer.steps.Retrieve`.

Caching
=======
Embedders internally perform caching to disk, so if you embed the same text multiple
times, the embedder will only embed the text once and then cache the results for
future runs.
"""

from .embedder import Embedder
from .openai_embedder import OpenAIEmbedder
from .parallel_embedder import ParallelEmbedder
from .sentence_transformers_embedder import SentenceTransformersEmbedder
from .together_embedder import TogetherEmbedder

__all__ = [
    "Embedder",
    "OpenAIEmbedder",
    "SentenceTransformersEmbedder",
    "TogetherEmbedder",
    "ParallelEmbedder",
]
