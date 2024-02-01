from functools import cached_property
from typing import Callable

from ._litellm import LiteLLM
from .llm import DEFAULT_BATCH_SIZE


class PaLM(LiteLLM):
    def __init__(
        self,
        model_name: str,
        api_key: None | str = None,
        retry_on_fail: bool = True,
        cache_folder_path: None | str = None,
        **kwargs,
    ):
        super().__init__(
            model_name=model_name,
            api_key=api_key,
            retry_on_fail=retry_on_fail,
            cache_folder_path=cache_folder_path,
            **kwargs,
        )
        self._model_name_prefix = "palm/"

    def _run_batch(
        self,
        max_length_func: Callable[[list[str]], int],
        inputs: list[str],
        max_new_tokens: None | int = None,
        temperature: float = 1.0,
        top_p: float = 0.0,
        n: int = 1,
        stop: None | str | list[str] = None,
        repetition_penalty: None | float = None,
        logit_bias: None | dict[int, float] = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        seed: None | int = None,
        **kwargs,
    ) -> list[str] | list[list[str]]:  # pragma: no cover
        assert (
            repetition_penalty is None
        ), f"`repetition_penalty` is not supported for {type(self).__name__}"
        return super()._run_batch(
            max_length_func=max_length_func,
            inputs=inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            n=n,
            stop=stop,
            repetition_penalty=repetition_penalty,
            logit_bias=logit_bias,
            batch_size=batch_size,
            seed=seed,
            **kwargs,
        )

    @cached_property
    def model_card(self) -> None | str:
        return "https://ai.google/static/documents/palm2techreport.pdf"

    @cached_property
    def license(self) -> None | str:
        return "https://policies.google.com/terms"

    @cached_property
    def citation(self) -> None | list[str]:
        citations = []
        citations.append(
            """@article{Anil2023PaLM2T,
  title={PaLM 2 Technical Report},"""
            """author={Rohan Anil and Andrew M. Dai and Orhan Firat and Melvin Johnson and Dmitry Lepikhin and Alexandre Tachard Passos and Siamak Shakeri and Emanuel Taropa and Paige Bailey and Z. Chen and Eric Chu and J. Clark and Laurent El Shafey and Yanping Huang and Kathleen S. Meier-Hellstern and Gaurav Mishra and Erica Moreira and Mark Omernick and Kevin Robinson and Sebastian Ruder and Yi Tay and Kefan Xiao and Yuanzhong Xu and Yujing Zhang and Gustavo Hernandez Abrego and Junwhan Ahn and Jacob Austin and Paul Barham and Jan A. Botha and James Bradbury and Siddhartha Brahma and Kevin Michael Brooks and Michele Catasta and Yongzhou Cheng and Colin Cherry and Christopher A. Choquette-Choo and Aakanksha Chowdhery and C Cr{\'e}py and Shachi Dave and Mostafa Dehghani and Sunipa Dev and Jacob Devlin and M. C. D'iaz and Nan Du and Ethan Dyer and Vladimir Feinberg and Fan Feng and Vlad Fienber and Markus Freitag and Xavier Garc{\'i}a and Sebastian Gehrmann and Lucas Gonz{\'a}lez and Guy Gur-Ari and Steven Hand and Hadi Hashemi and Le Hou and Joshua Howland and An Ren Hu and Jeffrey Hui and Jeremy Hurwitz and Michael Isard and Abe Ittycheriah and Matthew Jagielski and Wen Hao Jia and Kathleen Kenealy and Maxim Krikun and Sneha Kudugunta and Chang Lan and Katherine Lee and Benjamin Lee and Eric Li and Mu-Li Li and Wei Li and Yaguang Li and Jun Yu Li and Hyeontaek Lim and Han Lin and Zhong-Zhong Liu and Frederick Liu and Marcello Maggioni and Aroma Mahendru and Joshua Maynez and Vedant Misra and Maysam Moussalem and Zachary Nado and John Nham and Eric Ni and Andrew Nystrom and Alicia Parrish and Marie Pellat and Martin Polacek and Alex Polozov and Reiner Pope and Siyuan Qiao and Emily Reif and Bryan Richter and Parker Riley and Alexandra Ros and Aurko Roy and Brennan Saeta and Rajkumar Samuel and Renee Marie Shelby and Ambrose Slone and Daniel Smilkov and David R. So and Daniela Sohn and Simon Tokumine and Dasha Valter and Vijay Vasudevan and Kiran Vodrahalli and Xuezhi Wang and Pidong Wang and Zirui Wang and Tao Wang and John Wieting and Yuhuai Wu and Ke Xu and Yunhan Xu and Lin Wu Xue and Pengcheng Yin and Jiahui Yu and Qiaoling Zhang and Steven Zheng and Ce Zheng and Wei Zhou and Denny Zhou and Slav Petrov and Yonghui Wu},"""  # noqa: B950
            """journal={ArXiv},
  year={2023},
  volume={abs/2305.10403},
  url={https://api.semanticscholar.org/CorpusID:258740735}
}""".strip()
        )
        return citations


__all__ = ["PaLM"]
