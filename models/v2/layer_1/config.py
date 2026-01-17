from dataclasses import dataclass

@dataclass
class Config:
    train_csv: str
    val_csv: str
    test_csv: str

    text_col: str = "text"
    label_col: str = "label"

    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    max_length: int = 256
    batch_size: int = 64

    latent_dim: int = 128
    hidden_dim: int = 512
    dropout: float = 0.1

    epochs: int = 5
    lr: float = 2e-4
    err_kind: str = "mse"
    target_fpr: float = 0.01
    seed: int = 42
    out_dir: str = "ae_out"
