class StepOutputError(Exception):
    pass


class StepOutputTypeError(TypeError, StepOutputError):
    def __init__(self, message):
        if message:
            super().__init__(
                "Error processing dataset, make sure all values for each output of the"
                " dataset are of the same Python type/shape."
                f" Detailed error: {message.replace('struct', 'dict')}"
            )
        else:
            super().__init__(
                "Error processing dataset, make sure all values for each output of the"
                " dataset are of the same Python type/shape."
            )
