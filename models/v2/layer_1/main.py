from config import Config
from dataset import load_csv, TextDataset, collate_fn
from models.embedder import HFEmbedder
from models.autoencoder import MLPAutoEncoder
from train import train_ae
from eval import score_dataset
from torch.utils.data import DataLoader
import torch

cfg = Config(
    train_csv="train.csv",
    val_csv="val.csv",
    test_csv="test.csv",
)

device = torch.device("cuda")

embedder = HFEmbedder(cfg.model_name).to(device)
ae = MLPAutoEncoder(embedder.hidden_size, cfg.latent_dim, cfg.hidden_dim, cfg.dropout).to(device)

train_texts, _ = load_csv(cfg.train_csv, cfg.text_col, cfg.label_col)
train_ds = TextDataset(train_texts)
train_loader = DataLoader(train_ds, batch_size=cfg.batch_size,
                          collate_fn=collate_fn(embedder.tokenizer, cfg.max_length))

train_ae(ae, embedder, train_loader, device, cfg.lr, cfg.epochs)