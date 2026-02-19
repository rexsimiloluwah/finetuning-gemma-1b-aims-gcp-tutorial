import argparse
import json
import os
from datasets import load_dataset


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    dataset = load_dataset(
        "databricks/databricks-dolly-15k",
        split="train",
    )

    eval_subset = dataset.shuffle(seed=42).select(range(args.num_samples))

    output_path = os.path.join(args.output_dir, args.filename)

    with open(output_path, "w") as f:
        for item in eval_subset:
            json.dump(
                {
                    "instruction": item["instruction"],
                    "response": item["response"],
                },
                f,
            )
            f.write("\n")

    print(f"✅ Evaluation file written to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="data/eval")
    parser.add_argument("--filename", default="dolly_eval_500.jsonl")
    parser.add_argument("--num_samples", type=int, default=500)

    args = parser.parse_args()
    main(args)