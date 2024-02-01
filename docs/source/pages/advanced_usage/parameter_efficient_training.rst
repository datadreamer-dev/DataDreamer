Parameter-Efficient Training
#######################################################

DataDreamer makes setting up parameter-efficient training simple.
You can pass a :py:class:`~peft.PeftConfig` to the ``peft_config`` argument of a class
like :py:class:`~datadreamer.trainers.TrainHFFineTune` to enable parameter-efficient training.