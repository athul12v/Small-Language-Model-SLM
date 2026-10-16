from huggingface_hub import snapshot_download
from pathlib import Path

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
LOCAL_DIR = Path.home() / "slm-finetuning" / "models" / "qwen2.5-0.5b-base"

LOCAL_DIR.mkdir(parents=True, exist_ok=True)

print(f"Model: {MODEL_ID}")
print(f"Download directory: {LOCAL_DIR}")
print("Downloading model files from Hugging Face...")

snapshot_download(
    repo_id=MODEL_ID,
    local_dir=str(LOCAL_DIR),
)

print("\nDownload completed successfully!")
print(f"Model saved at: {LOCAL_DIR}")
