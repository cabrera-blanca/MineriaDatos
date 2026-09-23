"""Partición reproducible train/test del histórico (estratificada por Churn)."""
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split


def load_params(path: str = "params.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def split_data(df: pd.DataFrame, target: str, test_size: float, seed: int):
    return train_test_split(df, test_size=test_size, random_state=seed, stratify=df[target])


def main() -> None:
    p = load_params()["data"]
    df = pd.read_csv(p["raw_path"])
    train, test = split_data(df, p["target"], p["test_size"], p["seed"])
    train.to_csv(p["train_path"], index=False)
    test.to_csv(p["test_path"], index=False)
    print(f"train={train.shape} test={test.shape} churn_train={(train[p['target']]=='Yes').mean():.3f} "
          f"churn_test={(test[p['target']]=='Yes').mean():.3f}")


if __name__ == "__main__":
    main()
