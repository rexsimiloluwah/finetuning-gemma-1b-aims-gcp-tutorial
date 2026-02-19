import json
import math
import os
import torch
import wandb
from tqdm import tqdm
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.utils.device_utils import get_device

load_dotenv()


def repetition_rate(text: str, ngram: int = 3) -> float:
    """
    Compute n-gram repetition rate for generated text.
    """
    tokens = text.split()
    if len(tokens) < ngram:
        return 0.0

    ngrams = [
        tuple(tokens[i : i + ngram])
        for i in range(len(tokens) - ngram + 1)
    ]

    return 1.0 - (len(set(ngrams)) / len(ngrams))


def evaluate(
    model_path: str,
    eval_file: str,
    max_new_tokens: int = 128,
    max_eval_samples: int = 50,
):
    """
    Evaluate a fine-tuned Gemma chat model using perplexity
    and repetition rate on a Dolly-derived evaluation set.
    """
    device = get_device()

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    model.to(device)
    model.eval()

    wandb.init(
        project=os.environ.get("WANDB_PROJECT", "gemma-finetune"),
        name=f"eval-{os.path.basename(model_path)}",
    )

    examples = []
    with open(eval_file, "r") as f:
        for line in f:
            examples.append(json.loads(line))
            if len(examples) >= max_eval_samples:
                break

    total_loss = 0.0
    total_tokens = 0
    repetition_scores = []

    for example in tqdm(examples, desc="Evaluating"):

        # Perplexity (teacher-forced, chat-formatted)
        messages = [
            {"role": "user", "content": example["instruction"]},
            {"role": "assistant", "content": example["response"]},
        ]

        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_tensors="pt",
            return_dict=True,
        ).to(device)

        with torch.no_grad():
            outputs = model(
                input_ids=inputs["input_ids"],
                labels=inputs["input_ids"],
            )

        total_loss += outputs.loss.item() * inputs["input_ids"].numel()
        total_tokens += inputs["input_ids"].numel()

        # Generation for repetition rate
        gen_inputs = tokenizer.apply_chat_template(
            [{"role": "user", "content": example["instruction"]}],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to(device)

        gen_outputs = model.generate(
            input_ids=gen_inputs["input_ids"],
            max_new_tokens=max_new_tokens,
        )

        generated_text = tokenizer.decode(
            gen_outputs[0][gen_inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True,
        )

        repetition_scores.append(repetition_rate(generated_text))

    perplexity = math.exp(total_loss / total_tokens)
    avg_repetition = sum(repetition_scores) / len(repetition_scores)

    print(f"Perplexity: {perplexity:.4f}")
    print(f"Average repetition rate: {avg_repetition:.4f}")

    wandb.log({
        "eval/perplexity": perplexity,
        "eval/repetition_rate": avg_repetition,
    })

    wandb.finish()