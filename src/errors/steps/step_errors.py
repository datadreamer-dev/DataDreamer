class StepOutputTypeError(TypeError):
    def __init__(self, message):
        if message:
            super().__init__(
                "All values for each output of a dataset must be of the same"
                f" Python type, detailed error: {message.replace('struct', 'dict')}"
            )
        else:
            super().__init__(
                "All values for each output of a dataset must be of the same"
                " Python type."
            )
