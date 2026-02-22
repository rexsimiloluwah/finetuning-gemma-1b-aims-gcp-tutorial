import argparse
from src.inference import load_model, generate, interactive_mode


def main():
    parser = argparse.ArgumentParser(description="Run inference on a fine-tuned Gemma 1B LoRA model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to LoRA checkpoint")
    parser.add_argument("--base_model", type=str, default="google/gemma-3-1b-it", help="Base model name")
    parser.add_argument("--instruction", type=str, default=None, help="Single instruction to run (optional)")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling")
    args = parser.parse_args()

    print(f"Loading model from {args.model_path}...")
    model, tokenizer, device = load_model(args.model_path, args.base_model)
    print("Model loaded.\n")

    if args.instruction:
        response = generate(
            model,
            tokenizer,
            device,
            args.instruction,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        print(f"Instruction: {args.instruction}")
        print(f"Response: {response}")
    else:
        interactive_mode(model, tokenizer, device, args)

if __name__ == "__main__":
    main()