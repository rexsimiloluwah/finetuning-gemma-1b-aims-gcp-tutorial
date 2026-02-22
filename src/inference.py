import torch
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from src.utils.device_utils import get_device

load_dotenv()


def load_model(model_path: str, base_model_name: str):
    """
    Load a fine-tuned LoRA model for inference.
    """
    device = get_device()

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
    )

    model = PeftModel.from_pretrained(base_model, model_path)
    model = model.merge_and_unload()
    model.to(device)
    model.eval()

    return model, tokenizer, device


def generate(
    model,
    tokenizer,
    device,
    instruction: str,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
):
    """
    Generate a response for a given instruction.
    """
    messages = [{"role": "user", "content": instruction}]

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
        )

    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True,
    )

    return response


def interactive_mode(model, tokenizer, device, args):
    """
    Run an interactive prompt loop.
    """
    print("\n🤖 Gemma 1B Fine-tuned — Interactive Mode")
    print("Type your instruction and press Enter. Type 'quit' to exit.\n")

    while True:
        instruction = input("You: ").strip()
        if instruction.lower() in ("quit", "exit", "q"):
            print("Exiting.")
            break
        if not instruction:
            continue

        response = generate(
            model,
            tokenizer,
            device,
            instruction,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        print(f"\nModel: {response}\n")