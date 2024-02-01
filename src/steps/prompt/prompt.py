from ._prompt_base import _PromptBase


class Prompt(_PromptBase):
    "Runs a set of prompts against a :py:class:`~datadreamer.llms.LLM`."

    def setup(self):
        self._register_prompt_inputs()
        self._register_prompt_args()
        self._register_prompt_optional_args()
        self._register_prompt_outputs()

    def run(self):
        return self._run_prompts(args=self.args)


__all__ = ["Prompt"]
