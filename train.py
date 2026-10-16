import os
import torch

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)

# =====================================================
# 1. Configuration
# =====================================================

MODEL_PATH = "./models/qwen2.5-0.5b-base"
DATA_PATH = "./data/enterprise_slm_dataset.jsonl"
OUTPUT_DIR = "./outputs/enterprise-slm-adapter"

MAX_LENGTH = 512
NUM_EPOCHS = 3

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA is unavailable. Check your PyTorch installation."
    )

print("GPU:", torch.cuda.get_device_name(0))
print("GPU memory:", round(
    torch.cuda.get_device_properties(0).total_memory / 1024**3, 2
), "GiB")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =====================================================
# 2. Load tokenizer
# =====================================================

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    local_files_only=True,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

tokenizer.padding_side = "right"

# =====================================================
# 3. Load the model in 4-bit precision
# =====================================================

print("\nLoading model in 4-bit precision...")

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=quantization_config,
    dtype=torch.float16,
    device_map={"": 0},
    local_files_only=True,
)

model.config.use_cache = False

# =====================================================
# 4. Prepare model for QLoRA
# =====================================================

print("\nPreparing model for QLoRA...")

model = prepare_model_for_kbit_training(
    model,
    use_gradient_checkpointing=True,
)

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
# 5. Load the enterprise JSONL dataset
# =====================================================

print("\nLoading enterprise dataset...")

dataset = load_dataset(
    "json",
    data_files=DATA_PATH,
    split="train",
)

required_columns = {"instruction", "input", "output"}

if not required_columns.issubset(dataset.column_names):
    raise ValueError(
        f"Dataset must contain these fields: {required_columns}. "
        f"Found: {dataset.column_names}"
    )

dataset = dataset.filter(
    lambda row: (
        isinstance(row["instruction"], str)
        and isinstance(row["input"], str)
        and isinstance(row["output"], str)
        and bool(row["instruction"].strip())
        and bool(row["output"].strip())
    )
)

if len(dataset) < 2:
    raise ValueError("Not enough valid records to train.")

print("Valid records:", len(dataset))

# Reserve 10% for evaluation.
splits = dataset.train_test_split(
    test_size=0.1,
    seed=42,
)

train_dataset = splits["train"]
eval_dataset = splits["test"]

print("Training records:", len(train_dataset))
print("Evaluation records:", len(eval_dataset))

# =====================================================
# 6. Format and tokenize the examples
# =====================================================

def preprocess(example):
    user_content = (
        "Answer the employee's question using the supplied "
        "enterprise context. If the context does not contain "
        "the answer, say that the information is insufficient.\n\n"
        f"Question:\n{example['instruction']}\n\n"
        f"Enterprise context:\n{example['input']}"
    )

    # Prompt tokens: these will not contribute to the loss.
    prompt_messages = [
        {
            "role": "user",
            "content": user_content,
        }
    ]

    prompt_ids = tokenizer.apply_chat_template(
    prompt_messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=False,
)

    # Full conversation, including the expected answer.
    full_messages = [
        {
            "role": "user",
            "content": user_content,
        },
        {
            "role": "assistant",
            "content": example["output"],
        },
    ]

    full_ids = tokenizer.apply_chat_template(
    full_messages,
    tokenize=True,
    add_generation_prompt=False,
    return_dict=False,
)

    # Avoid training on the user's question and context.
    # Train on the assistant's answer instead.
    labels = list(full_ids)
    prompt_length = min(len(prompt_ids), len(labels))

    labels[:prompt_length] = [-100] * prompt_length

    # Reserve room for at least one answer token.
    if len(full_ids) > MAX_LENGTH:
        full_ids = full_ids[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]

    return {
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": labels,
    }


print("\nTokenizing dataset...")

train_dataset = train_dataset.map(
    preprocess,
    remove_columns=train_dataset.column_names,
)

eval_dataset = eval_dataset.map(
    preprocess,
    remove_columns=eval_dataset.column_names,
)

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
    output_dir=OUTPUT_DIR,

    num_train_epochs=NUM_EPOCHS,

    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,

    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_steps=3,
    weight_decay=0.01,

    fp16=True,
    gradient_checkpointing=True,

    optim="adamw_torch",

    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="steps",
    logging_steps=5,

    save_total_limit=2,
    load_best_model_at_end=True,

    report_to="none",
    seed=42,
    data_seed=42,
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
    data_collator=data_collator,
)

print("\nStarting enterprise SLM fine-tuning...")
print("Monitor GPU memory in another terminal with:")
print("watch -n 1 nvidia-smi")

trainer.train()

# =====================================================
# 9. Save the trained adapter
# =====================================================

print("\nSaving trained adapter...")

trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("\nFine-tuning completed!")
print("Adapter saved to:", OUTPUT_DIR)
