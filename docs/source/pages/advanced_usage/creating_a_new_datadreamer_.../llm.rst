Creating a new LLM
#######################################################

To create a new DataDreamer LLM class to support a new LLM library or API service, you will want to subclass
the :py:class:`~datadreamer.llms.LLM` class. You can see example implementations of various LLMs by clicking on the
``[source]`` links on the :doc:`LLMs <../../../datadreamer.llms>` page. These may be helpful as reference implementations.

Contributing
============

If you would like to contribute the new LLM class you created to DataDreamer for others to use, see the :doc:`Contributing <../../../pages/contributing>` page.
If applicable, please ensure your implementation includes model metadata, such as a link to the model card, the model's license, and the model's citation
information.