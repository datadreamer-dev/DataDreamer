"""
:py:class:`Retriever` objects help retrieve texts based on a set of queries. 
All retrievers derive from the :py:class:`Retriever` base class.

.. tip::

   Instead of using :py:meth:`~Retriever.run` directly, use a
   :py:class:`step <datadreamer.steps>` that takes a :py:class:`Retriever` as an ``args``
   argument such as :py:class:`~datadreamer.steps.Retrieve` and
   :py:class:`~datadreamer.steps.RAGPrompt`. Some other steps like and
   :py:class:`~datadreamer.steps.FewShotPromptWithRetrieval`
   use retrievers internally.

Caching
=======
Retrievers typically initially build an index once and cache the index to disk. 
Retrievers additionally internally perform caching to disk, so if you retrieve results
for the same query multiple times, the retriever will only retrieve results for the
query once and then cache the results for future runs.
"""

from .embedding_retriever import EmbeddingRetriever
from .parallel_retriever import ParallelRetriever
from .retriever import Retriever

__all__ = ["Retriever", "EmbeddingRetriever", "ParallelRetriever"]
