import os
import argparse

import numpy as np
import torch
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error


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
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.task == "short":
        args.pred_len = 90
        data_path = "data/saved_dataset/dataset_short.npz"
    else:
        args.pred_len = 365
        data_path = "data/saved_dataset/dataset_long.npz"

    data = np.load(data_path)
    X_test = data["X_test"]
    Y_test = data["Y_test"]

    scaler = np.load("data/saved_dataset/scaler_info.npz")
    y_mean = scaler["Y_mean"]
    y_std = scaler["Y_std"]

    model = load_model(args).to(device)
    ckpt = f"results/{args.model}_{args.task}/seed{args.seed}/best_model.pth"
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    X_test_tensor = torch.FloatTensor(X_test).to(device)
    with torch.no_grad():
        pred = model(X_test_tensor)

    pred = pred.cpu().numpy()

    # 逆标准化还原真实值
    pred_real = pred * y_std + y_mean
    true_real = Y_test * y_std + y_mean

    # 计算指标
    mae = mean_absolute_error(true_real.flatten(), pred_real.flatten())
    mse = mean_squared_error(true_real.flatten(), pred_real.flatten())

    save_dir = f"results/{args.model}_{args.task}/seed{args.seed}"

    with open(os.path.join(save_dir, "metrics.txt"), "w") as f:
        f.write(f"MAE={mae}\n")
        f.write(f"MSE={mse}\n")

    print("\n========== RESULT ==========")
    print("Model :", args.model)
    print("Task  :", args.task)
    print("Seed  :", args.seed)
    print("MAE   :", mae)
    print("MSE   :", mse)

    # 画三张图：第一个样本、中间样本、最后一个样本

    num_samples = len(true_real)
    sample_indices = [0, num_samples // 2, num_samples - 1]
    sample_names = ["first", "middle", "last"]

    for idx, name in zip(sample_indices, sample_names):
        plt.figure(figsize=(12, 5))
        plt.plot(true_real[idx], label="Ground Truth", linewidth=1.5)
        plt.plot(pred_real[idx], label="Prediction", linewidth=1.5, linestyle="--")
        plt.xlabel("Prediction Horizon")
        plt.ylabel("Global Active Power (kW)")
        plt.title(f"{args.model} - {args.task} - {name} sample (seed {args.seed})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"prediction_curve_{name}.png"))
        plt.close()


if __name__ == "__main__":
    main()