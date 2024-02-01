from typing import Callable

import pytest

from ....steps import Step
from ....steps.step import _INTERNAL_TEST_KEY


@pytest.fixture
def create_test_step() -> Callable[..., Step]:
    def _create_test_step(
        name="my-step",
        inputs=None,
        args=None,
        outputs=None,
        output_names=None,
        setup=None,
        **kwargs,
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

        setattr(TestStep, _INTERNAL_TEST_KEY, True)

        return TestStep(name, inputs=inputs, args=args, outputs=outputs, **kwargs)

    return _create_test_step
