import pathlib

max_seq_len: int = 256
max_batch_size: int = 2
llama_model_name: str = "llama-2-7b-chat"

ROOT_DIR = pathlib.Path(__file__).parent.parent.parent
ckpt_dir = str((ROOT_DIR / llama_model_name).resolve())
tokenizer_path = str((ROOT_DIR / "tokenizer.model").resolve())
