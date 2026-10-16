import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import PeftModel

# --------------------------------------------------
# Configuration
# --------------------------------------------------

BASE_MODEL = "./models/qwen2.5-0.5b-base"
ADAPTER_PATH = "./outputs/enterprise-slm-adapter"

# --------------------------------------------------
# Check GPU
# --------------------------------------------------

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable. Check your PyTorch installation.")

print("GPU:", torch.cuda.get_device_name(0))

# --------------------------------------------------
# Load tokenizer
# --------------------------------------------------

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL,
    local_files_only=True,
)

# --------------------------------------------------
# Load base model in 4-bit precision
# --------------------------------------------------

print("\nLoading base model in 4-bit precision...")

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=quantization_config,
    device_map="auto",
    local_files_only=True,
)

# --------------------------------------------------
# Load trained adapter
# --------------------------------------------------

print("\nLoading trained enterprise adapter...")

model = PeftModel.from_pretrained(
    base_model,
    ADAPTER_PATH,
    is_trainable=False,
)

model.eval()

print("\nFine-tuned enterprise SLM loaded successfully!")

# --------------------------------------------------
# Generate an answer
# --------------------------------------------------

def ask_enterprise_model(question, context):
    user_content = (
        "Answer the employee's question using the supplied "
        "enterprise context. If the context does not contain "
        "the answer, say that the information is insufficient.\n\n"
        f"Question:\n{question}\n\n"
        f"Enterprise context:\n{context}"
    )

    messages = [
        {
            "role": "user",
            "content": user_content,
        }
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )

    # Move inputs to the model's input device.
    inputs = inputs.to(model.get_input_embeddings().weight.device)

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=120,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Display only the newly generated answer.
    prompt_length = inputs["input_ids"].shape[1]
    generated_tokens = output[0][prompt_length:]

    answer = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    )

    return answer.strip()


# --------------------------------------------------
# Interactive enterprise Q&A
# --------------------------------------------------

print("\n" + "=" * 60)
print("ENTERPRISE SLM DEMO")
print("Type 'exit' to quit.")
print("=" * 60)

while True:
    question = input("\nEmployee question: ").strip()

    if question.lower() in {"exit", "quit"}:
        break

    if not question:
        continue

    context = input("Enterprise context: ").strip()

    if not context:
        print("Please provide enterprise context.")
        continue

    answer = ask_enterprise_model(question, context)

    print("\nSLM answer:")
    print(answer)