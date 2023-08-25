from llama import Llama
from llama.application import settings
from llama.generation import Dialog, ChatPrediction

generator = Llama.build(
    ckpt_dir=settings.ckpt_dir,
    tokenizer_path=settings.tokenizer_path,
    max_seq_len=settings.max_seq_len,
    max_batch_size=settings.max_batch_size,
)


def get_num_tokens(text: str) -> int:
    return len(generator.tokenizer.encode(text, bos=False, eos=False))


def get_first_n_tokens(text: str, n: int) -> str:
    tokens = generator.tokenizer.encode(text, bos=False, eos=False)
    return generator.tokenizer.decode(tokens[:n])


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
