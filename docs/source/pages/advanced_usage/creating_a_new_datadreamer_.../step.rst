Creating a new Step
#######################################################

Creating a new step in DataDreamer is useful for creating custom logic to transform or process input data into some new output data.
It is something you will commonly want to do in a workflow to implement a new or custom technique.

When you create and run a new step in a DataDreamer session, you immediately get the benefits of the DataDreamer library such as reusability,
`caching of outputs <../../../datadreamer.steps.html#caching>`_, enhanced logging,
`parallelizability <../../../pages/advanced_usage/parallelization/running_steps_in_parallel.html>`_, easy interoperability with the other
`steps <../../../datadreamer.steps.html#types-of-steps>`_ and
`trainers <../../../datadreamer.trainers.html>`_ available in DataDreamer, and
`automatic data card generation <../../../datadreamer.steps.html#data-card-generation>`_.

You can create a new step by subclassing the :py:class:`~datadreamer.steps.Step` class and
implementing the :py:meth:`~datadreamer.steps.Step.setup` and :py:meth:`~datadreamer.steps.Step.run` methods:

.. code-block:: python

	from datadreamer.steps import Step

	class MyNewStep(Step):
		def setup(self):
			# Register inputs, arguments, outputs, and data card information here

		def run(self):
			# Implement your custom data processing / transformation logic here

Implementing :py:meth:`~datadreamer.steps.Step.setup`
=====================================================

The :py:meth:`~datadreamer.steps.Step.setup` method registers what inputs and arguments the step will accept, and what outputs it will return. It also allows you to register
data card information for the step, that ultimately is used by DataDreamer to
`automatically generate data cards <../../../datadreamer.steps.html#data-card-generation>`_.

Registering inputs, arguments, and outputs
------------------------------------------

You can use :py:meth:`self.register_input() <datadreamer.steps.Step.register_input>` to register the name of each input that the step will
accept. Inputs will be provided as :py:class:`~datadreamer.datasets.OutputDatasetColumn` or
:py:class:`~datadreamer.datasets.OutputIterableDatasetColumn` objects. 

You can use :py:meth:`self.register_arg() <datadreamer.steps.Step.register_arg>` to register the name of each argument that the step will
accept. Arguments can be provided of any type.

You can use :py:meth:`self.register_output() <datadreamer.steps.Step.register_output>` to register the name of each output that the step will
produce. These outputs must be returned by your :py:meth:`~datadreamer.steps.Step.run` method implementation.

Registering data card information
---------------------------------

You can use :py:meth:`self.register_data_card() <datadreamer.steps.Step.register_data_card>` to register various data card information where
``data_card_type`` can be a one of the :py:class:`~datadreamer.steps.DataCardType` types, and ``data_card_value`` can be then information you wish
to add to the data card for that :py:class:`~datadreamer.steps.DataCardType`.

Implementing :py:meth:`~datadreamer.steps.Step.run`
===================================================

The :py:meth:`~datadreamer.steps.Step.run` method is where you implement your custom data processing / transformation logic using the input data and arguments
requested in :py:meth:`~datadreamer.steps.Step.setup`. Your implementation of :py:meth:`~datadreamer.steps.Step.run` must also return outputs
that correspond to the outputs registered in :py:meth:`~datadreamer.steps.Step.setup`.

Accessing inputs and arguments
------------------------------

You can access the inputs and arguments provided to the step by accessing the :py:attr:`self.inputs <datadreamer.steps.Step.inputs>` and :py:attr:`self.args <datadreamer.steps.Step.args>` dictionaries, respectively.

Storing persistent data
-----------------------

If you need a folder to store persistent data during :py:meth:`~datadreamer.steps.Step.run`, you can use the :py:meth:`self.get_run_output_folder_path() <datadreamer.steps.Step.get_run_output_folder_path>` method.

Returning outputs
-----------------

You can ``return`` a dataset of outputs from :py:meth:`~datadreamer.steps.Step.run` corresponding with the output column names registered in :py:meth:`~datadreamer.steps.Step.setup`. DataDreamer will automatically
convert the return value to an :py:class:`~datadreamer.datasets.OutputDataset` object and make it available on the :py:attr:`~datadreamer.steps.Step.output`
attribute of the step.

.. dropdown:: Valid Return Formats

	DataDreamer is very flexible in what you can return as outputs, and you can return an output dataset in any of the following ways:

	- You can return a dictionary of lists, where each key is the name of an output column, and each value is a list of values for that output column.
	- You can return a list of dictionaries, where each list item is a row of data, and each dictionary key is the name of an output column, and each dictionary value is the value for that output column.
	- You can return a list of tuples, where each list item is a row of data, and each tuple item is the value for each output column in the order they were registered.
	- You can return a Hugging Face :py:class:`~datasets.Dataset` object or :py:class:`~datasets.IterableDataset` object.
	- ... other data structures are also supported, DataDreamer will try to understand what you are returning and convert it to an appropriate dataset of outputs.

	.. note::

		If any of your output columns contain a value that is not a primitive Python type (``bool``, ``str``, ``float``, ``int``, ``list``, etc.) you may get a type error stating
		that the value is not valid since it cannot be serialized. If this happens, you can pickle the values of the column by using
		:py:meth:`self.pickle() <datadreamer.steps.Step.pickle>`. This will allow you to return arbitrary Python types. DataDreamer will automatically unpickle the values
		when the output dataset is accessed. You can also use :py:meth:`self.unpickle() <datadreamer.steps.Step.unpickle>` to manually unpickle values if needed.
 
Returning outputs lazily
^^^^^^^^^^^^^^^^^^^^^^^^

If you want to return outputs lazily to make your step run as a `lazy step <../../../datadreamer.steps.html#lazy-steps>`_, you can return a generator
function that will ``yield`` a single row of data at a time instead and wrap the function with :py:class:`~datadreamer.steps.LazyRows` before returning it. If you want
your generator function to ``yield`` a batch of rows at a time, you can wrap the function with :py:class:`~datadreamer.steps.LazyRowBatches` instead.

Updating the progress indicator
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

DataDreamer can keep the user updated on the progress of your step if you periodically update the progress by setting the
:py:attr:`self.progress <datadreamer.steps.Step.progress>` attribute to a value between 0 and 1. If you are returning outputs lazily,
DataDreamer will automatically update the progress based on the number of rows yielded so far.

Running steps within steps
--------------------------

If you want to run other steps inside :py:meth:`~datadreamer.steps.Step.run`, then you must subclass the
:py:class:`~datadreamer.steps.SuperStep` class instead of the :py:class:`~datadreamer.steps.Step` class.

Contributing
============
You may want to contribute the new step class you created to DataDreamer for others to use, especially if it is a reusable
technique. See the :doc:`Contributing <../../../pages/contributing>` page for how to contribute your step that others may benefit
from using. If applicable, please ensure your implementation includes data card metadata, such as a link to the model/data
cards used, any license information, and any citation information.