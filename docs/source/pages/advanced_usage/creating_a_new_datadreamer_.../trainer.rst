Creating a new Trainer
#######################################################

To create a new DataDreamer trainer class to support a new training library or training technique, you will want to subclass
the :py:class:`~datadreamer.trainers.Trainer` class. You can see example implementations of various trainers by clicking on the
``[source]`` links on the :doc:`Trainers <../../../datadreamer.trainers>` page. These may be helpful as reference implementations.

Contributing
============

If you would like to contribute the new trainer class you created to DataDreamer for others to use, see the :doc:`Contributing <../../../pages/contributing>` page.
If applicable, please ensure your implementation includes appropriate metadata, such as a link to the model card of the model being trained, the model's license, and
the model and training technique's citation information.