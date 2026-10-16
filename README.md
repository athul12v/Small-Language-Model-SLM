# SLM for Enterprises – Fine-Tuning and Implementation

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-ee4c2c?logo=pytorch&logoColor=white)
![Hugging Face](https://img.shields.io/badge/🤗-Transformers-yellow)
![QLoRA](https://img.shields.io/badge/Fine--Tuning-QLoRA-green)
![PEFT](https://img.shields.io/badge/PEFT-LoRA-blueviolet)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-ff4b4b?logo=streamlit&logoColor=white)
![Inference](https://img.shields.io/badge/Inference-Local%20GPU-blue)
![License](https://img.shields.io/badge/License-To%20Be%20Determined-lightgrey)

## Overview

**SLM for Enterprises** is a Small Language Model (SLM) fine-tuned to answer employee questions using enterprise-specific information, such as company policies, procurement procedures, finance guidelines, and engineering documentation.

The project uses **Qwen2.5-0.5B** and applies **QLoRA fine-tuning** to adapt the model's response behavior to enterprise question-and-answer data.

The goal is to build a lightweight AI assistant that can run locally on a compatible GPU and provide answers grounded in relevant enterprise information.

## How It Works

The system consists of two main stages.

### 1. Fine-Tuning the SLM

The base Qwen2.5-0.5B model is fine-tuned using enterprise-specific data stored in JSONL format.

Each training example contains:

- **Instruction:** The employee's question or topic.
- **Input:** The relevant enterprise information.
- **Output:** The expected answer.

QLoRA combines 4-bit quantization with Low-Rank Adaptation (LoRA) to reduce memory requirements during fine-tuning.

During training, the model learns to follow instructions and generate answers in the desired format.

The trained LoRA adapter is saved separately from the base model.

### 2. Enterprise AI Assistant

After fine-tuning, a Streamlit application provides a simple chat interface for employees to ask questions.

When a user submits a question, the system follows these steps:

1. **Receive the question:** The employee enters a question through the chat interface.
2. **Retrieve relevant information:** A keyword-based search identifies relevant records from the enterprise dataset.
3. **Prepare the context:** The retrieved information is combined with the employee's question.
4. **Generate an answer:** The fine-tuned SLM generates a response using the supplied enterprise information.
5. **Display the response:** The answer appears in the chat interface, with an option to view the supporting context.

If the system cannot find sufficiently relevant information, it returns a fallback message directing the employee to the appropriate company team or documentation.

## Technology Stack

| Technology | Purpose |
|---|---|
| Qwen2.5-0.5B | Base language model |
| QLoRA | Memory-efficient fine-tuning |
| 4-bit NF4 | Model quantization |
| Hugging Face Transformers | Model loading and training |
| PEFT | LoRA adapter configuration |
| Hugging Face Datasets | Dataset loading and processing |
| PyTorch | Deep learning framework |
| Streamlit | Interactive chat interface |
| CUDA | GPU-accelerated training and inference |

## Key Benefits

- **Lightweight model:** Uses a small language model instead of a large, resource-intensive model.
- **Memory-efficient fine-tuning:** QLoRA reduces GPU memory requirements.
- **Enterprise-specific behavior:** Fine-tuning adapts the model to enterprise question-and-answer tasks.
- **Context-based responses:** Retrieves relevant enterprise information to support generated answers.
- **Local execution:** Supports local inference without requiring a hosted LLM API.
- **Extensible architecture:** Can be extended with semantic search, document ingestion, and additional enterprise data sources.

## Current Limitations

- **Keyword-based retrieval:** The current implementation uses keyword matching rather than semantic search. It may miss relevant information when a question uses different wording from the source documents.
- **Model capabilities:** A 0.5B-parameter model may struggle with complex reasoning and nuanced questions.
- **Data dependency:** Answer quality depends on the accuracy and completeness of the enterprise dataset.
- **Potential hallucinations:** The model may generate incorrect information, even when instructed to use only the supplied context.

Responses should be validated before use in business-critical workflows.

## Future Improvements

- Implement semantic search using sentence embeddings.
- Integrate a vector database such as FAISS or Chroma.
- Add source citations to generated answers.
- Improve evaluation using a separate test dataset.
- Add enterprise document upload and ingestion.
- Implement authentication and role-based access control.
- Add response quality checks and monitoring.

## Conclusion

**SLM for Enterprises** demonstrates how a small language model can be fine-tuned using QLoRA and integrated into an enterprise question-answering application.

By combining parameter-efficient fine-tuning, enterprise information retrieval, and a simple chat interface, the project provides a foundation for developing lightweight, locally running AI assistants tailored to organizational needs.

---
