from pathlib import Path
import argparse
import pandas as pd


def load_table(path: Path) -> pd.DataFrame:
    suffixes = "".join(path.suffixes).lower()

    if suffixes.endswith(".csv"):
        return pd.read_csv(path)

    if suffixes.endswith(".tsv"):
        return pd.read_csv(path, sep="\t")

    if suffixes.endswith(".jsonl") or suffixes.endswith(".jsonl.gz"):
        return pd.read_json(path, lines=True)

    if suffixes.endswith(".json"):
        return pd.read_json(path)

    if suffixes.endswith(".parquet"):
        return pd.read_parquet(path)

    raise ValueError(f"Unsupported file type: {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    args = ap.parse_args()

    path = Path(args.path)
    df = load_table(path)

    print("=" * 100)
    print("Path:", path)
    print("Rows:", len(df))
    print("Columns:", df.columns.tolist())
    print("\nFirst three rows:")
    print(df.head(3).to_string())

    print("\nPossible label columns:")
    for col in df.columns:
        name = col.lower()
        if any(token in name for token in [
            "label", "gold", "target", "class", "slant", "political"
        ]):
            print(f"\n--- {col} ---")
            print(df[col].value_counts(dropna=False).head(30))


if __name__ == "__main__":
    main()
