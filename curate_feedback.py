"""
curate_feedback.py - Converts Human Feedback into LoRA Training Records.
Reads data/hitl_feedback.jsonl -> Validates -> Appends to data/enterprise_slm_dataset.jsonl
"""

import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
FEEDBACK_FILE = BASE_DIR / "data" / "hitl_feedback.jsonl"
TRAIN_FILE = BASE_DIR / "data" / "enterprise_slm_dataset.jsonl"


def process_feedback_to_training():
    if not FEEDBACK_FILE.exists():
        print(f"No feedback file found at: {FEEDBACK_FILE}")
        print("Submit feedback or corrections in the Web UI first!")
        return 0

    curated_records = []
    with open(FEEDBACK_FILE, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except Exception:
                continue

            # Look for records where a human provided a corrected answer
            chosen = item.get("chosen_answer") or item.get("corrected_answer")
            query = item.get("query")
            context = item.get("context", "")

            if chosen and query and len(chosen.strip()) > 3:
                record = {
                    "instruction": query.strip(),
                    "input": context.strip() if context.strip() else "Verified company documentation.",
                    "output": chosen.strip(),
                }
                curated_records.append(record)

    if not curated_records:
        print("No human corrections found in data/hitl_feedback.jsonl to convert.")
        return 0

    # Append to the training dataset
    with open(TRAIN_FILE, "a", encoding="utf-8") as f:
        for r in curated_records:
            f.write(json.dumps(r) + "\n")

    print("=" * 60)
    print(f">> Successfully extracted {len(curated_records)} human correction(s)!")
    print(f">> Appended directly to: {TRAIN_FILE}")
    print("=" * 60)
    print("Next step: Run 'python train.py' to fine-tune your SLM with these new human corrections.")
    return len(curated_records)


if __name__ == "__main__":
    process_feedback_to_training()

