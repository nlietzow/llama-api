import pathlib

from pydantic_settings import BaseSettings

ROOT_DIR = pathlib.Path(__file__).parent.parent.parent


class Settings(BaseSettings):
    model_name: str = "llama-2-7b-chat"
    max_seq_len: int = 512
    max_batch_size: int = 4

    @property
    def ckpt_dir(self) -> str:
        path = ROOT_DIR / self.model_name
        if not path.exists():
            raise ValueError(f"Model {self.model_name} not found")

        return str(path.resolve())

    @property
    def tokenizer_path(self) -> str:
        path = ROOT_DIR / "tokenizer.model"
        if not path.exists():
            raise ValueError(f"Tokenizer not found")

        return str(path.resolve())


settings = Settings()
