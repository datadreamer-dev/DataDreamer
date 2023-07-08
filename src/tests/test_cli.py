import pytest

from ..__cli__ import _main


class TestCli:
    @pytest.mark.skip(reason="skipping because slow")
    def test_hello_world(self, cli_runner):
        result = cli_runner.invoke(_main, ["hello-world"])
        assert result.output.count("My cell phone number is") == 4
