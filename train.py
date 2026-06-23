import os
import csv
import argparse
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class TimeSeriesDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.FloatTensor(X)
        self.Y = torch.FloatTensor(Y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


def load_model(args):
    if args.model == "lstm":
        from models.lstm import Model
    elif args.model == "patch":
        from models.patch import Model
    elif args.model == "transformer":
        from models.transformer import Model
    else:
        raise ValueError(f"Unknown model: {args.model}")

    return Model(
        input_dim=13,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        pred_len=args.pred_len,
        dropout=args.dropout
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.task == "short":
        data_path = "data/saved_dataset/dataset_short.npz"
        args.pred_len = 90
    else:
        data_path = "data/saved_dataset/dataset_long.npz"
        args.pred_len = 365

    data = np.load(data_path)

    X_train_full = data["X_train"]
    Y_train_full = data["Y_train"]

    split_idx = int(len(X_train_full) * 0.8)
    X_train = X_train_full[:split_idx]
    Y_train = Y_train_full[:split_idx]
    X_val = X_train_full[split_idx:]
    Y_val = Y_train_full[split_idx:]

    train_dataset = TimeSeriesDataset(X_train, Y_train)
    val_dataset = TimeSeriesDataset(X_val, Y_val)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False
    )

    model = load_model(args).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    save_dir = f"results/{args.model}_{args.task}/seed{args.seed}"
    os.makedirs(save_dir, exist_ok=True)

    log_file = os.path.join(save_dir, "train_log.csv")
    with open(log_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss"])

    best_loss = float("inf")
    patience = 10
    counter = 0

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0

        for X, Y in train_loader:
            X = X.to(device)
            Y = Y.to(device)

            optimizer.zero_grad()
            pred = model(X)
            loss = criterion(pred, Y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0

        with torch.no_grad():
            for X, Y in val_loader:
                X = X.to(device)
                Y = Y.to(device)
                pred = model(X)
                loss = criterion(pred, Y)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        print(f"Epoch [{epoch+1}] Train={train_loss:.6f} Val={val_loss:.6f}")

        with open(log_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, train_loss, val_loss])

        if val_loss < best_loss:
            best_loss = val_loss
            counter = 0
            torch.save(
                model.state_dict(),
                os.path.join(save_dir, "best_model.pth")
            )
        else:
            counter += 1

        if counter >= patience:
            print("Early Stopping")
            break

    print(f"\nBest Validation Loss: {best_loss:.6f}")


if __name__ == "__main__":
    main()