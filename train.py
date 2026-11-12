import json
import os
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model

# =====================================================
# 1. Configuration
# =====================================================

BASE_DIR = Path(__file__).resolve().parent
LOCAL_MODEL_PATH = BASE_DIR / "models" / "qwen2.5-0.5b-base"
FALLBACK_LOCAL_MODEL_PATH = Path.home() / "slm-finetuning" / "models" / "qwen2.5-0.5b-base"
HF_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

DATA_PATH = BASE_DIR / "data" / "enterprise_slm_dataset.jsonl"
OUTPUT_DIR = BASE_DIR / "outputs" / "enterprise-slm-adapter"

MAX_LENGTH = 512
NUM_EPOCHS = 3
SEED = 42


class TokenizedSFTDataset(Dataset):
    """Lightweight PyTorch Dataset wrapping pre-tokenized SFT examples."""

    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def resolve_model_source():
    """Return (model_path_or_id, local_files_only) using local weights if available."""
    for candidate in (LOCAL_MODEL_PATH, FALLBACK_LOCAL_MODEL_PATH):
        if candidate.exists() and (candidate / "config.json").exists():
            return str(candidate), True
    print(
        f"Local model not found at {LOCAL_MODEL_PATH}. "
        f"Falling back to Hugging Face Hub ({HF_MODEL_ID})..."
    )
    return HF_MODEL_ID, False


def load_jsonl_records(data_path: Path):
    """Load and validate instruction/input/output records from a JSONL dataset."""
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {data_path}")

    required_columns = {"instruction", "input", "output"}
    records = []

    with open(data_path, "r", encoding="utf-8") as f:
        for line_num, raw_line in enumerate(f, 1):
            line = raw_line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not required_columns.issubset(row.keys()):
                raise ValueError(
                    f"Line {line_num}: Dataset must contain fields {required_columns}. "
                    f"Found: {set(row.keys())}"
                )
            if (
                isinstance(row["instruction"], str)
                and isinstance(row["input"], str)
                and isinstance(row["output"], str)
                and row["instruction"].strip()
                and row["output"].strip()
            ):
                records.append(row)

    if len(records) < 2:
        raise ValueError("Not enough valid records to train (minimum 2 required).")

    return records


def _extract_ids(template_output) -> list[int]:
    """Normalize apply_chat_template output into a flat list of token IDs."""
    if isinstance(template_output, dict) or hasattr(template_output, "get"):
        ids = template_output["input_ids"]
        return list(ids[0]) if ids and isinstance(ids[0], list) else list(ids)
    return list(template_output)


def preprocess_record(example, tokenizer, max_length: int):
    """Format a record with chat template and mask prompt tokens from loss."""
    user_content = (
        "Answer the employee's question using the supplied "
        "enterprise context. If the context does not contain "
        "the answer, say that the information is insufficient.\n\n"
        f"Question:\n{example['instruction']}\n\n"
        f"Enterprise context:\n{example['input']}"
    )

    # Prompt tokens: excluded from loss computation via -100 masking.
    prompt_messages = [{"role": "user", "content": user_content}]
    prompt_ids = _extract_ids(
        tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=False,
        )
    )

    # Full conversation, including the expected assistant answer.
    full_messages = [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": example["output"]},
    ]
    full_ids = _extract_ids(
        tokenizer.apply_chat_template(
            full_messages,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=False,
        )
    )

    labels = list(full_ids)
    prompt_length = min(len(prompt_ids), len(labels))
    labels[:prompt_length] = [-100] * prompt_length

    if len(full_ids) > max_length:
        full_ids = full_ids[:max_length]
        labels = labels[:max_length]

    # Guard against NaN loss when truncation removes all assistant answer tokens.
    if not any(label != -100 for label in labels):
        return None

    return {
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": labels,
    }


def main():
    use_cuda = torch.cuda.is_available()

    if use_cuda:
        print("GPU:", torch.cuda.get_device_name(0))
        print(
            "GPU memory:",
            round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2),
            "GiB",
        )
    else:
        print("CUDA is unavailable. Falling back to CPU training (float32).")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model_source, local_only = resolve_model_source()

    # =====================================================
    # 2. Load tokenizer
    # =====================================================

    print(f"\nLoading tokenizer from {model_source}...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_source,
        local_files_only=local_only,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "right"

    # =====================================================
    # 3. Load the model (FP16 on GPU, FP32 on CPU)
    # =====================================================

    precision_label = "16-bit (FP16)" if use_cuda else "32-bit (FP32 CPU)"
    print(f"\nLoading model in {precision_label} for LoRA...")

    model_kwargs = {
        "dtype": torch.float16 if use_cuda else torch.float32,
        "local_files_only": local_only,
    }
    if use_cuda:
        model_kwargs["device_map"] = {"": 0}

    model = AutoModelForCausalLM.from_pretrained(model_source, **model_kwargs)
    model.config.use_cache = False

    # =====================================================
    # 4. Prepare model for LoRA
    # =====================================================

    print("\nPreparing model for LoRA...")

    if use_cuda:
        model.enable_input_require_grads()

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # =====================================================
    # 5. Load and split the enterprise JSONL dataset
    # =====================================================

    print("\nLoading enterprise dataset...")
    records = load_jsonl_records(DATA_PATH)
    print("Valid records:", len(records))

    # Reserve 10% for evaluation.
    rng = random.Random(SEED)
    rng.shuffle(records)

    eval_count = max(1, int(len(records) * 0.1))
    eval_records = records[:eval_count]
    train_records = records[eval_count:]

    print("Training records:", len(train_records))
    print("Evaluation records:", len(eval_records))

    # =====================================================
    # 6. Format and tokenize the examples
    # =====================================================

    print("\nTokenizing dataset...")
    train_samples = [
        sample
        for row in train_records
        if (sample := preprocess_record(row, tokenizer, MAX_LENGTH)) is not None
    ]
    eval_samples = [
        sample
        for row in eval_records
        if (sample := preprocess_record(row, tokenizer, MAX_LENGTH)) is not None
    ]

    if not train_samples or not eval_samples:
        raise ValueError("No usable tokenized samples remained after preprocessing.")

    train_dataset = TokenizedSFTDataset(train_samples)
    eval_dataset = TokenizedSFTDataset(eval_samples)

    # =====================================================
    # 7. Configure training
    # =====================================================

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=None,
        padding=True,
        label_pad_token_id=-100,
        pad_to_multiple_of=8,
    )

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_steps=3,
        weight_decay=0.01,
        fp16=use_cuda,
        use_cpu=not use_cuda,
        gradient_checkpointing=use_cuda,
        gradient_checkpointing_kwargs={"use_reentrant": False} if use_cuda else None,
        dataloader_pin_memory=use_cuda,
        remove_unused_columns=False,
        optim="adamw_torch",
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=5,
        save_total_limit=2,
        load_best_model_at_end=True,
        report_to="none",
        seed=SEED,
        data_seed=SEED,
        dataloader_num_workers=0,
    )

    # =====================================================
    # 8. Start fine-tuning
    # =====================================================

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
    )

    print("\nStarting enterprise SLM fine-tuning...")
    if use_cuda:
        print("Monitor GPU memory in another terminal with:")
        print("watch -n 1 nvidia-smi")

    trainer.train()

    # =====================================================
    # 9. Save the trained adapter
    # =====================================================

    print("\nSaving trained adapter...")
    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))

    print("\nFine-tuning completed!")
    print("Adapter saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
