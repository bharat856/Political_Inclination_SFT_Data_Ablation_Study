import argparse
import itertools

from dolci_intervention_loader import (
    VALID_CONDITIONS,
    expected_row_count,
    load_dolci_intervention,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--condition",
        required=True,
        choices=VALID_CONDITIONS,
    )
    parser.add_argument(
        "--show-examples",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--count-all",
        action="store_true",
        help="Stream through the full corpus and verify its row count.",
    )
    args = parser.parse_args()

    dataset = load_dolci_intervention(args.condition)

    print("\nExpected rows:", f"{expected_row_count(args.condition):,}")

    print("\nExamples:")
    for i, row in enumerate(
        itertools.islice(dataset, args.show_examples)
    ):
        print("\n" + "=" * 100)
        print("Example:", i)
        print("row_id:", row["row_id"])
        print("split:", row["split"])
        print("text preview:", row["text"][:500])

    if args.count_all:
        dataset = load_dolci_intervention(args.condition)
        actual = sum(1 for _ in dataset)
        expected = expected_row_count(args.condition)

        print("\nActual rows:", f"{actual:,}")
        print("Expected rows:", f"{expected:,}")

        if actual != expected:
            raise RuntimeError(
                f"Count mismatch: expected {expected:,}, got {actual:,}"
            )

        print("Row-count verification passed.")


if __name__ == "__main__":
    main()
