import torch
from tqdm import tqdm

def reconstruction_error(x, x_hat, kind="mse"):
    if kind == "mse":
        return ((x_hat - x) ** 2).mean(dim=1)
    else:
        return (x_hat - x).abs().mean(dim=1)

def train_ae(ae, embedder, loader, device, lr, epochs, err_kind="mse"):
    ae.train()
    embedder.eval()
    opt = torch.optim.AdamW(ae.parameters(), lr=lr)

    for ep in range(epochs):
        for batch in tqdm(loader):
            x = embedder.encode(
                batch.input_ids.to(device),
                batch.attention_mask.to(device),
            )
            x_hat, _ = ae(x)
            loss = reconstruction_error(x, x_hat, err_kind).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
