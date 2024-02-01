from ..__cli__ import _main


class TestCli:
    def test_help(self, cli_runner):
        result = cli_runner.invoke(_main, ["--help"])
        assert result.output.count("Show this message and exit.") == 1
