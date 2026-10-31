import argparse
from dolci_intervention_loader import load_dolci_intervention

parser = argparse.ArgumentParser()
parser.add_argument("--rank", type=int, required=True)
parser.add_argument("--world-size", type=int, default=4)
parser.add_argument("--n", type=int, default=20)
args = parser.parse_args()

ds = load_dolci_intervention(
    "full",
    shuffle_buffer_size=10000,
    seed=42,
    rank=args.rank,
    world_size=args.world_size,
)

print(f"RANK={args.rank}")
for i, row in enumerate(ds):
    print(row["split"], row["row_id"])
    if i + 1 >= args.n:
        break
