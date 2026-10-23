# Enterprise SLM with AI Harness

A fine-tuned **Small Language Model (SLM)** wrapped in a simple, human-readable **AI Harness**.

The harness provides reliable guardrails around the SLM:
- **Input Validation**: Rejects empty inputs and prompt injections.
- **Missing Info Detection**: Asks polite follow-up questions instead of guessing or hallucinating.
- **Context Management**: Remembers previous facts across multiple conversation turns.
- **Enterprise Retrieval**: Grounds answers strictly in verified enterprise documents.
- **Structured Output**: Always returns a predictable JSON response format.

---

## Simple File Structure

```
Small-Language-Model-SLM/
│
├── harness.py              # The entire AI Harness in ONE simple, readable file
├── app.py                  # Web server hosting the HTML/CSS chat UI
├── chat.py                 # Interactive terminal CLI chat
├── evaluate.py             # Simple 6-scenario evaluation test
│
├── template/               # Simple Web UI (Self-contained)
│   └── index.html          # Clean HTML UI with inline styling and script
│
├── data/
│   └── enterprise_slm_dataset.jsonl  # 100 enterprise Q&A records
└── README.md
```

---

## How to Run This Project

### 1. Start the Web App (HTML/CSS UI)
```bash
python app.py
```
Open **`http://localhost:8000`** in your browser.

---

### 2. Run the Terminal CLI Chat
```bash
python chat.py
```
Ask questions directly in your terminal. Type `exit` to quit.

---

### 3. Run the Evaluation Benchmark
```bash
python evaluate.py
```
Runs automated checks for valid queries, missing fields, ambiguity, contradictions, unanswerable queries, and prompt injection defense.

---

## Standard JSON Output Schema

Every request processed by `harness.process(...)` or `POST /api/chat` returns:

```json
{
  "status": "success | missing_information | clarification_required | insufficient_evidence | validation_error",
  "answer": "Answer grounded in enterprise documents",
  "missing_fields": [],
  "follow_up_question": null,
  "sources": [
    {
      "instruction": "Question topic",
      "input": "Document content",
      "score": 1.73
    }
  ],
  "metadata": {
    "latency_ms": 5.2,
    "tokens": 40
  }
}
```
