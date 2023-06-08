from ..__cli__ import _main


class TestCli:
    def test_hello_world(self, cli_runner):
        result = cli_runner.invoke(_main, ["hello-world", "-t", "3"])
        assert result.output.count("Hello world!") == 3
