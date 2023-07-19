from typing import Callable

import pytest

from ....steps import Step


@pytest.fixture
def create_test_step() -> Callable[..., Step]:
    def _create_test_step(
        name="my-step",
        inputs=None,
        args=None,
        outputs=None,
        output_names=None,
        setup=None,
    ) -> Step:
        if output_names is None:
            output_names = []

        class TestStep(Step):
            def setup(self):
                if isinstance(output_names, str):
                    self.register_output(output_names)
                else:
                    for o in output_names:
                        self.register_output(o)
                if setup is not None:
                    setup(self)

        return TestStep(name, inputs=inputs, args=args, outputs=outputs)

    return _create_test_step
