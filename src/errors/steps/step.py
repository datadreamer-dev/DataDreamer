class StepOutputError(Exception):
    pass


class StepOutputTypeError(TypeError, StepOutputError):
    def __init__(self, message: str):
        if message:
            super().__init__(
                "Error processing dataset, make sure all values for each output of the"
                " dataset are of the same Python type/shape. If you need more"
                " flexibility you can pickle your data using the .pickle() method on"
                " a Step object. Data will automatically be un-pickled when read."
                f" Detailed error: {message.replace('struct', 'dict')}"
            )
