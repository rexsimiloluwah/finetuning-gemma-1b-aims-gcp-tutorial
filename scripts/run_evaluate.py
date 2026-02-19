import argparse
from src.evaluate import evaluate

def main():
    parser = argparse.ArgumentParser(description="Evaluate Gemma 1B LoRA model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--eval_file", type=str, default="data/eval/eval_prompts.jsonl", help="Path to eval JSONL file")
    parser.add_argument("--max_new_tokens", type=int, default=128, help="Max new tokens to generate")
    parser.add_argument("--max_eval_samples", type=int, default=50, help="Max number of eval examples to use")
    args = parser.parse_args()

    evaluate(
        model_path=args.model_path,
        eval_file=args.eval_file,
        max_new_tokens=args.max_new_tokens,
        max_eval_samples=args.max_eval_samples,
    )

if __name__ == "__main__":
    main()