from pathlib import Path
from datasets import load_dataset

DATASETS = [
    "allenai/Dolci-Instruct-SFT-7B",
    "allenai/Dolci-Think-SFT-7B",
]

OUT_DIR = Path("data/raw/olmo3_dolci_sft")
OUT_DIR.mkdir(parents=True, exist_ok=True)

for name in DATASETS:
    print("=" * 120)
    print("Loading dataset:", name)

    ds = load_dataset(name)

    print(ds)

    safe_name = name.split("/")[-1]
    dataset_out_dir = OUT_DIR / safe_name
    dataset_out_dir.mkdir(parents=True, exist_ok=True)

    for split in ds:
        print("\nSPLIT:", split)
        print("Rows:", len(ds[split]))
        print("Columns:", ds[split].column_names)

        print("\nFirst example:")
        print(ds[split][0])

        # Save a small preview for inspection
        preview_path = dataset_out_dir / f"{split}_preview.jsonl"
        ds[split].select(range(min(20, len(ds[split])))).to_json(str(preview_path), lines=True)
        print("Saved preview:", preview_path)

    print("Finished:", name)

print("\nAll datasets loaded and cached successfully.")
