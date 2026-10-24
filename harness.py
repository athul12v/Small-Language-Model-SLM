"""
harness.py - Simple, Human-Readable AI Harness for Enterprise SLMs.

Wraps a fine-tuned Small Language Model (Qwen2.5-0.5B) with safety guardrails:
1. Input Validation: Catches empty inputs and prompt injections.
2. Missing Info Detection: Asks polite follow-up questions instead of guessing.
3. Multi-Turn Context: Remembers previous facts and user answers.
4. Enterprise Retrieval: Pulls verified documents from the dataset.
5. Structured Output: Returns a clean, consistent JSON dictionary.
"""

import os
import re
import json
import time
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "enterprise_slm_dataset.jsonl"
MODEL_PATH = BASE_DIR / "models" / "qwen2.5-0.5b-base"
ADAPTER_PATH = BASE_DIR / "outputs" / "enterprise-slm-adapter"


# =====================================================================
# 1. Enterprise Knowledge Retriever (Keyword Search)
# =====================================================================
class KnowledgeRetriever:
    """Searches the enterprise JSONL dataset for documents matching the user's question."""

    STOP_WORDS = {
        "the", "is", "a", "an", "of", "to", "for", "in", "on", "our", "we", "i",
        "what", "where", "how", "can", "do", "does", "and", "or", "my", "with",
        "about", "are", "be", "it", "at", "as", "by", "this", "that"
    }

    def __init__(self, data_file: Optional[Path] = None):
        self.records: List[Dict[str, str]] = []
        path = data_file or DATA_PATH
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            self.records.append(json.loads(line))
                        except Exception:
                            pass

    def _words(self, text: str) -> set:
        return {
            w for w in re.findall(r"\b[a-zA-Z0-9_\-]+\b", text.lower())
            if len(w) > 1 and w not in self.STOP_WORDS
        }

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        query_words = self._words(query)
        if not query_words or not self.records:
            return []

        scored = []
        for r in self.records:
            doc_text = f"{r.get('instruction', '')} {r.get('input', '')}"
            overlap = query_words.intersection(self._words(doc_text))
            if overlap:
                score = len(overlap) / (len(query_words) ** 0.5)
                scored.append((score, r))

        scored.sort(key=lambda x: x[0], reverse=True)
        if not scored or scored[0][0] < 1.0:
            return []

        best_score = scored[0][0]
        return [dict(rec, score=round(score, 2)) for score, rec in scored[:top_k] if score >= best_score * 0.4]


# =====================================================================
# 2. SLM Model Wrapper (Qwen2.5-0.5B + PEFT, with Demo Fallback)
# =====================================================================
class ModelBackend:
    """Runs inference on the fine-tuned SLM, or provides a grounded fallback if weights/GPU are absent."""

    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.is_real_model = False
        self._try_load_model()

    def _try_load_model(self):
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
            from peft import PeftModel

            if torch.cuda.is_available() and MODEL_PATH.exists() and ADAPTER_PATH.exists():
                print(f"Loading Qwen2.5-0.5B on GPU: {torch.cuda.get_device_name(0)}...")
                self.tokenizer = AutoTokenizer.from_pretrained(str(ADAPTER_PATH), local_files_only=True)
                qconfig = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
                base = AutoModelForCausalLM.from_pretrained(
                    str(MODEL_PATH),
                    quantization_config=qconfig,
                    device_map="auto",
                    local_files_only=True,
                )
                self.model = PeftModel.from_pretrained(base, str(ADAPTER_PATH), local_files_only=True)
                self.model.eval()
                self.is_real_model = True
                print("Model loaded successfully.")
            else:
                print("Model weights or CUDA GPU not detected. Using Demo Mode (retrieving real enterprise records).")
        except Exception as e:
            print(f"PyTorch model not loaded ({e}). Running in Demo Mode.")

    def generate(self, messages: List[Dict[str, str]], fallback_answer: str = "") -> Tuple[str, int]:
        """Generates text from the SLM, returning (answer_text, token_count)."""
        if not self.is_real_model or not self.model or not self.tokenizer:
            # Fallback answer based on the retrieved documents
            return (fallback_answer or "Information is documented in company records."), 40

        import torch
        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        prompt_len = inputs["input_ids"].shape[1]

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=180,
                do_sample=False,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        new_tokens = outputs[0][prompt_len:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        return text, len(new_tokens)


# =====================================================================
# 3. The AI Harness
# =====================================================================
class Harness:
    """
    Main harness orchestrator.
    Handles validation, missing information detection, conversation history,
    document retrieval, and structured response generation.
    """

    # Prompt injection patterns
    INJECTION_PATTERNS = [
        r"(?i)\bignore\s+(all\s+)?(previous|prior|above)\s+(instructions|prompts|rules)\b",
        r"(?i)\byou\s+are\s+now\s+(unrestricted|in\s+developer\s+mode|dan)\b",
        r"(?i)\bsystem\s+prompt\s*:\s*",
        r"(?i)\boutput\s+your\s+(initial|system)\s+instructions\b",
        r"(?i)\breveal\s+(the\s+)?(hidden\s+|secret\s+|system\s+)*(prompt|instructions|rules)\b",
        r"(?i)<\s*system\s*>",
    ]

    # Ambiguous short question hints
    AMBIGUOUS_TERMS = {
        "policy": "Which company policy are you inquiring about (e.g., Travel, PTO, Open-Source, or Data Governance)?",
        "leave": "What type of leave are you asking about (e.g., Parental Leave, Sick Leave, or Paid Time Off)?",
        "reimbursement": "Which type of expense do you need information for (e.g., Meal Per Diem, Tuition, or Hardware)?",
    }

    def __init__(self):
        self.retriever = KnowledgeRetriever()
        self.backend = ModelBackend()
        # In-memory multi-turn session storage: { session_id: { "facts": {}, "history": [] } }
        self.sessions: Dict[str, Dict[str, Any]] = {}

    def get_session(self, session_id: str) -> Dict[str, Any]:
        if session_id not in self.sessions:
            self.sessions[session_id] = {"facts": {}, "history": []}
        return self.sessions[session_id]

    def reset_session(self, session_id: str):
        if session_id in self.sessions:
            del self.sessions[session_id]

    # --- Guardrail 1: Input Validation & Sanitization ---
    def validate_input(self, query: str) -> Tuple[bool, Optional[str]]:
        if not isinstance(query, str) or not query.strip():
            return False, "Query cannot be empty."
        if len(query.strip()) < 2:
            return False, "Query is too short."

        for pattern in self.INJECTION_PATTERNS:
            if re.search(pattern, query):
                return False, "Security rejection: prompt injection detected."

        return True, None

    def sanitize_text(self, text: str) -> str:
        # Strip malicious delimiters and redact passwords/keys
        cleaned = text.replace("<|im_start|>", "").replace("<|im_end|>", "")
        cleaned = re.sub(r"(?i)(api[_-]?key|token|password)\s*[:=]\s*['\"]?\w+['\"]?", r"\1: [REDACTED]", cleaned)
        return cleaned.strip()

    # --- Guardrail 2: Missing Information & Ambiguity Check ---
    def check_missing_info(
        self,
        query: str,
        required_fields: List[str],
        session: Dict[str, Any],
        provided_fields: Dict[str, Any],
    ) -> Tuple[Optional[str], List[str], Optional[str]]:
        """
        Returns (status, missing_fields, follow_up_question) if clarification is needed.
        """
        # Combine stored facts with newly provided fields
        available = dict(session["facts"])
        available.update(provided_fields)

        # 1. Check required fields
        missing = [f for f in required_fields if f not in available or not str(available[f]).strip()]
        if missing:
            return (
                "missing_information",
                missing,
                f"To answer accurately, could you please provide your {', '.join(missing)}?",
            )

        # 2. Check contradiction (e.g. contractor vs full-time)
        q_lower = query.lower()
        if "contractor" in q_lower and "full_time" in q_lower:
            return (
                "clarification_required",
                ["employment_status"],
                "There is a conflict between 'contractor' and 'full_time' in your query. Could you clarify your employment status?",
            )

        # 3. Check ambiguity for overly short questions
        words = set(re.findall(r"\b\w+\b", q_lower))
        if len(words) <= 5:
            for term, question in self.AMBIGUOUS_TERMS.items():
                if term in words and not any(term in str(v).lower() for v in available.values()):
                    return (
                        "clarification_required",
                        [f"{term}_topic"],
                        question,
                    )

        return None, [], None

    # --- Main Pipeline Execution ---
    def process(
        self,
        query: str,
        session_id: str = "default",
        provided_fields: Optional[Dict[str, Any]] = None,
        required_fields: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Runs query through validation, missing-info checks, retrieval, and inference.
        Returns a clean dictionary matching the target schema.
        """
        start_time = time.perf_counter()
        session = self.get_session(session_id)
        provided_fields = provided_fields or {}
        required_fields = required_fields or []

        # 1. Validate Input
        is_valid, error_msg = self.validate_input(query)
        if not is_valid:
            return {
                "status": "validation_error",
                "answer": f"Input validation rejected the request: {error_msg}",
                "missing_fields": [],
                "follow_up_question": None,
                "sources": [],
                "metadata": {"latency_ms": round((time.perf_counter() - start_time) * 1000, 1)},
            }

        clean_query = self.sanitize_text(query)

        # Update session facts
        for k, v in provided_fields.items():
            if str(v).strip():
                session["facts"][k] = v

        # 2. Check for Missing Info or Ambiguity
        status, missing_fields, follow_up = self.check_missing_info(
            clean_query, required_fields, session, provided_fields
        )
        if status:
            return {
                "status": status,
                "answer": None,
                "missing_fields": missing_fields,
                "follow_up_question": follow_up,
                "sources": [],
                "metadata": {"latency_ms": round((time.perf_counter() - start_time) * 1000, 1)},
            }

        # 3. Retrieve Supporting Enterprise Documents
        sources = self.retriever.search(clean_query)

        # If follow-up turn in multi-turn session has no direct keywords, combine with original question
        if not sources and session["history"]:
            original_q = session["history"][0].get("user", "")
            if original_q:
                sources = self.retriever.search(f"{original_q} {clean_query}")

        # If still no documents, fail safely
        if not sources:
            return {
                "status": "insufficient_evidence",
                "answer": "I couldn't find sufficiently relevant information in the enterprise knowledge base. Please consult the appropriate company team.",
                "missing_fields": [],
                "follow_up_question": None,
                "sources": [],
                "metadata": {"latency_ms": round((time.perf_counter() - start_time) * 1000, 1)},
            }

        # 4. Build Context with Strict Provenance Tags
        context_blocks = "\n\n".join([
            f"<doc_{i+1}>\nTopic: {s.get('instruction','')}\nDetails: {s.get('input','')}\n</doc_{i+1}>"
            for i, s in enumerate(sources)
        ])

        system_prompt = (
            "You are SLM Enterprise Assistant. Answer employee questions using strictly "
            "the provided verified documents. If the documents do not contain the answer, say "
            "that the information is insufficient. Do not guess or invent policies."
        )

        user_content = (
            f"<enterprise_knowledge>\n{context_blocks}\n</enterprise_knowledge>\n\n"
            f"<user_question>\n{clean_query}\n</user_question>"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        # 5. Generate Answer
        fallback = sources[0].get("output", "")
        answer, token_count = self.backend.generate(messages, fallback_answer=fallback)

        # Record turn in session history
        session["history"].append({"user": clean_query, "assistant": answer})

        latency = round((time.perf_counter() - start_time) * 1000, 1)

        return {
            "status": "success",
            "answer": answer,
            "missing_fields": [],
            "follow_up_question": None,
            "sources": sources,
            "metadata": {
                "latency_ms": latency,
                "tokens": token_count,
                "session_id": session_id,
            },
        }
