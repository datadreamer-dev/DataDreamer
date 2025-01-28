# type: ignore
# ruff: noqa

import json
import math
from collections import defaultdict
from functools import lru_cache
from typing import DefaultDict, Optional

import torch
from outlines_core.fsm.guide import (
    RegexGuide as CoreRegexGuide,
    create_states_mapping as uncached_create_states_mapping,
)
from outlines_core.fsm.json_schema import build_regex_from_schema
from transformers import LogitsProcessor, PreTrainedTokenizerBase


def cached_create_states_mapping(regex_string, tokenizer, *args, **kwargs):
    return uncached_create_states_mapping(regex_string, tokenizer, *args, **kwargs)


class RegexGuide(CoreRegexGuide):
    """
    Guide to generate text in the language of a regular expression.
    CoreRegexGuide with outlines cache
    """

    @classmethod
    def from_regex(cls, regex_string: str, tokenizer, **kwargs):
        return super().from_regex(
            regex_string,
            tokenizer,
            _create_states_mapping=cached_create_states_mapping,
            **kwargs,
        )


class _GrammarLogitProcessor(LogitsProcessor):
    fsm_state: DefaultDict[int, int]
    fsm: RegexGuide

    def __init__(self, tokenizer: Optional[PreTrainedTokenizerBase], grammar: str):
        self.tokenizer = _GrammarLogitProcessor._cached_adapt_tokenizer(tokenizer)
        self.fsm = _GrammarLogitProcessor._cached_compile_fsm(grammar, self.tokenizer)

    def __call__(self, logits: torch.Tensor, fsm_grammar_state: int):
        if fsm_grammar_state == -1 or self.fsm is None:
            return logits
        allowed_tokens = self.fsm.get_next_instruction(fsm_grammar_state).tokens
        mask = torch.full_like(logits, -math.inf)
        if allowed_tokens is not None:
            mask[:, allowed_tokens] = 0
        biased_scores = logits + mask
        return biased_scores

    def advance(self, next_token_id, fsm_grammar_state):
        return _GrammarLogitProcessor._advance(
            next_token_id, fsm_grammar_state, self.fsm
        )

    @staticmethod
    def _advance(next_token_id, fsm_grammar_state, fsm):
        if fsm_grammar_state == -1:
            return fsm_grammar_state
        return fsm.get_next_state(fsm_grammar_state, next_token_id)

    # TODO: move grammar compilation into the router
    @staticmethod
    @lru_cache(maxsize=32, typed=True)
    def _cached_compile_fsm(schema: str, tokenizer: Optional[PreTrainedTokenizerBase]):
        regex_str = schema
        fsm = RegexGuide.from_regex(regex_str, tokenizer)
        return fsm

    @staticmethod
    @lru_cache(maxsize=32, typed=True)
    def _cached_adapt_tokenizer(tokenizer):
        """Adapt tokenizer to work with the FSM.

        The API of Outlines tokenizers is slightly different to that of
        `transformers`. In addition we need to handle the missing spaces to
        Llama's tokenizer to be able to compile FSMs for this model.

        """
        tokenizer.vocabulary = tokenizer.get_vocab()
        tokenizer.special_tokens = set(tokenizer.all_special_tokens)

        def convert_token_to_string(token: str) -> str:
            from transformers.file_utils import SPIECE_UNDERLINE

            string = tokenizer.convert_tokens_to_string([token])

            # A hack to handle missing spaces to HF's Llama tokenizers
            if token.startswith(SPIECE_UNDERLINE) or token == "<0x20>":
                return " " + string

            return string

        tokenizer.convert_token_to_string = convert_token_to_string
        return tokenizer


class GrammarLogitProcessor:
    def __init__(
        self,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        grammar: str = "",
        fsm_grammar_state: Optional[int] = 0,
    ):
        self.grammar_processor = (
            _GrammarLogitProcessor(tokenizer, grammar) if grammar != "" else None
        )
        self.tokenizer = tokenizer
        self.grammar = grammar

        # Initialize FSM grammar states for each batch item
        self.fsm_grammar_states = defaultdict(lambda: fsm_grammar_state)

    def __call__(self, input_ids, scores):
        batch_size = scores.size(0)

        # Warp next scores with the grammar_processor
        if self.grammar_processor is not None:
            for i in range(batch_size):
                scores[i, :] = self.grammar_processor(
                    scores[i, :].unsqueeze(0), self.fsm_grammar_states[i]
                )[0]

        # Compute the log softmax of scores for the entire batch
        next_logprob = torch.log_softmax(scores, dim=-1)

        # Get the next ID with the highest score for each batch item
        next_ids = scores.argmax(dim=-1).view(batch_size, 1)

        # Advance grammar states for each batch item
        for i in range(batch_size):
            self.advance_grammar(i, next_ids[i].item())

        # Create a mask to set everything except next_ids to -inf
        mask = torch.full_like(scores, float("-inf"))
        mask.scatter_(
            dim=-1, index=next_ids, value=0.0
        )  # Set the score for next_id to 0 (log(1) = 0)

        next_logprob = mask  # Replace all scores with the mask

        return next_logprob

    def advance_grammar(self, batch_idx: int, next_id: int):
        if self.grammar_processor is not None:
            self.fsm_grammar_states[batch_idx] = self.grammar_processor.advance(
                next_id, self.fsm_grammar_states[batch_idx]
            )
        return self


class JSONLogitProcessor(GrammarLogitProcessor):
    def __init__(self, tokenizer, json_spec, whitespace_pattern=r"[\n ]*"):
        if not isinstance(json_spec, str):
            json_spec = json.dumps(json_spec)
        compiled_grammar = build_regex_from_schema(
            json_spec, whitespace_pattern=whitespace_pattern
        )
        super().__init__(tokenizer=tokenizer, grammar=compiled_grammar)


__all__ = ["GrammarLogitProcessor", "JSONLogitProcessor"]
