import gc

import torch
from typing_extensions import TypedDict

from llama.application import settings
from llama.application.retry_wrapper import retry_on_cuda_oom
from llama.generation import Dialog, ChatPrediction, Llama

gc.collect()
torch.cuda.empty_cache()

generator = Llama.build(
    ckpt_dir=settings.ckpt_dir,
    tokenizer_path=settings.tokenizer_path,
    max_seq_len=settings.max_seq_len,
    max_batch_size=settings.max_batch_size,
)


class TextWithNumTokens(TypedDict):
    text: str
    num_tokens: int


@retry_on_cuda_oom
def get_num_tokens(text: str) -> TextWithNumTokens:
    tokens = generator.tokenizer.encode(text, bos=False, eos=False)
    return TextWithNumTokens(text=text, num_tokens=len(tokens))


@retry_on_cuda_oom
def get_first_n_tokens(text: str, n: int) -> TextWithNumTokens:
    tokens = generator.tokenizer.encode(text, bos=False, eos=False)
    tokens = tokens[:n]
    text_max_n_tokens = generator.tokenizer.decode(tokens)
    return TextWithNumTokens(text=text_max_n_tokens, num_tokens=len(tokens))


@retry_on_cuda_oom
def predict(
    dialogs: list[Dialog],
    *,
    temperature: float,
    top_p: float,
    max_gen_len: int | None,
    logprobs: bool,
) -> list[ChatPrediction]:
    return generator.chat_completion(
        dialogs,
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
        logprobs=logprobs,
    )
