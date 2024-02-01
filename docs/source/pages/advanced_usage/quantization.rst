Quantization
#######################################################

DataDreamer makes setting up quantization simple. You can pass a
`quantization config object <https://huggingface.co/docs/transformers/main_classes/quantization>`_
to the ``quantization_config`` argument of a class like :py:class:`~datadreamer.llms.HFTransformers` or
:py:class:`~datadreamer.trainers.TrainHFFineTune` to enable quantization.