import os
import argparse

import numpy as np


def read_metric(path):
    result = {}
    with open(path, "r") as f:
        for line in f:
            key, value = line.strip().split("=")
            result[key] = float(value)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--task", type=str, required=True)
    args = parser.parse_args()

    maes = []
    mses = []
    seeds = [42, 43, 44, 45, 46]

    for seed in seeds:
        metric_path = f"results/{args.model}_{args.task}/seed{seed}/metrics.txt"
        result = read_metric(metric_path)
        maes.append(result["MAE"])
        mses.append(result["MSE"])

    mae_mean = np.mean(maes)
    mae_std = np.std(maes)
    mse_mean = np.mean(mses)
    mse_std = np.std(mses)

    print("\n========== FINAL ==========")
    print(f"MAE = {mae_mean:.4f} ± {mae_std:.4f}")
    print(f"MSE = {mse_mean:.4f} ± {mse_std:.4f}")

    save_dir = f"results/{args.model}_{args.task}"
    os.makedirs(save_dir, exist_ok=True)

    with open(os.path.join(save_dir, "summary.txt"), "w") as f:
        f.write(f"MAE={mae_mean:.6f}±{mae_std:.6f}\n")
        f.write(f"MSE={mse_mean:.6f}±{mse_std:.6f}\n")


if __name__ == "__main__":
    main()