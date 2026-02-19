import hydra
from omegaconf import DictConfig
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForSeq2Seq

from peft import LoraConfig, get_peft_model

from src.utils.data_utils import prepare_training_dataset, prepare_eval_dataset
from src.utils.device_utils import get_device

load_dotenv()

@hydra.main(version_base=None, config_path="configs", config_name="base")
def train(cfg: DictConfig) -> None:
    """
    Fine-tune Gemma-3-1B-IT using LoRA on Dolly-15k with Hydra configuration.
    """
    device = get_device()

    # Load and tokenize datasets
    train_dataset, tokenizer = prepare_training_dataset(cfg)
    eval_dataset = prepare_eval_dataset(cfg)

    # Load model
    model = AutoModelForCausalLM.from_pretrained(cfg.model.name)
    model.to(device)

    # Apply LoRA if enabled
    if cfg.lora.enabled:
        lora_config = LoraConfig(
            r=cfg.lora.r,
            lora_alpha=cfg.lora.alpha,
            lora_dropout=cfg.lora.dropout,
            target_modules=list(cfg.lora.target_modules), # Convert ListConfig to list
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)

    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=None,
        padding=True,
        pad_to_multiple_of=8,
    )

    # Training configuration
    training_args = TrainingArguments(
        output_dir=f"{cfg.training.output_dir}/{cfg.experiment_id}",
        logging_dir=f"{cfg.training.output_dir}/{cfg.experiment_id}/logs",
        per_device_train_batch_size=cfg.training.batch_size,
        learning_rate=cfg.training.learning_rate,
        num_train_epochs=cfg.training.num_epochs,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch",
        report_to="wandb",
        remove_unused_columns=False,
        bf16=True
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    trainer.train()


if __name__ == "__main__":
    train()