import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# ==========================================
# 1. Model paths
# ==========================================

BASE_MODEL = "./models/qwen2.5-0.5b-base"
ADAPTER_PATH = "./outputs/enterprise-slm-adapter"

# ==========================================
# 2. Device configuration
# ==========================================

if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU not detected.")

print("GPU:", torch.cuda.get_device_name(0))
print("Loading enterprise SLM...")

# ==========================================
# 3. Load tokenizer
# ==========================================

tokenizer = AutoTokenizer.from_pretrained(
    ADAPTER_PATH,
    local_files_only=True,
)

# ==========================================
# 4. Load base model in 4-bit precision
# ==========================================

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=quantization_config,
    device_map="auto",
    local_files_only=True,
)

# ==========================================
# 5. Apply trained adapter
# ==========================================

model = PeftModel.from_pretrained(
    base_model,
    ADAPTER_PATH,
    local_files_only=True,
)

model.eval()

print("\nEnterprise SLM loaded successfully!")
print("Type 'exit' to quit.\n")

# ==========================================
# 6. Chat loop
# ==========================================

SYSTEM_PROMPT = """
You are Apex Enterprise Assistant.

Answer employee questions clearly and professionally.
Use the enterprise context provided by the user.
If the context does not contain enough information,
state that the information is insufficient.
Do not invent company policies or facts.
"""

while True:
    question = input("Employee: ").strip()

    if question.lower() in {"exit", "quit"}:
        print("Goodbye!")
        break

    if not question:
        continue

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": question,
        },
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )

    inputs = {
        key: value.to(model.device)
        for key, value in inputs.items()
    }

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=180,
            do_sample=False,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Display only the newly generated answer.
    generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]

    answer = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    )

    print(f"\nApex Assistant: {answer}\n")