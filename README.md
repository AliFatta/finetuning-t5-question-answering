# Fine-Tuning T5-Base for Generative Question Answering (SQuAD)

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/🤗-Transformers-yellow.svg)](https://huggingface.co/docs/transformers)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A generative question answering system that fine-tunes **T5-base** on the **SQuAD v1.1 dataset** to generate textual answers directly from context, moving beyond traditional extractive QA approaches.

---

## 📥 Pre-trained Model Availability

The fine-tuned model generated in this project is saved locally after training (`./final_t5_squad_model`).

> **Note:** Due to repository size limitations and the file size of the T5 model (~850MB), model weights are not directly included in this repository. Inference can be performed immediately after training within the provided notebook or by loading the checkpoint path.

---

## 📔 Table of Contents

- [Purpose](#-purpose)
- [Project Overview](#-project-overview)
- [Dataset Preparation](#-dataset-preparation)
- [Tokenization](#-tokenization)
- [Model Configuration](#️-model-configuration)
- [Training Details](#️-training-details)
- [Inference and Usage](#-inference-and-usage)
- [Results and Observations](#-results-and-observations)
- [Repository Structure](#-repository-structure)
- [Installation & Setup](#️-installation--setup)
- [Conclusion](#-conclusion)

---

## 🎯 Purpose

This project focuses on fine-tuning the **T5-base (Text-to-Text Transfer Transformer)** model for **Generative Question Answering** using the **SQuAD v1.1 dataset**.

The main objective is to move beyond traditional *extractive* QA (like BERT, which predicts start/end indices) and instead train a Seq2Seq model to **generate** the textual answer directly given a context and a question.

---

## 🔍 Project Overview

### Generative Question Answering

Unlike classification or span-prediction tasks, this project treats Question Answering as a **Sequence-to-Sequence generation task**. The model receives a text input containing both the question and context, and it must output the correct answer string token-by-token.

**Key Differences from Extractive QA:**
- 🎯 **Extractive (BERT)**: Predicts start/end positions in the context
- ✨ **Generative (T5)**: Produces answer text token-by-token
- 🔄 **Flexibility**: Can paraphrase or summarize answers
- 💡 **Creativity**: Not limited to exact context spans

### T5 Model Architecture

**T5 (Text-to-Text Transfer Transformer)** is an Encoder-Decoder architecture. It is pre-trained on a multi-task mixture of unsupervised and supervised tasks, making it highly effective for transfer learning tasks where the input and output are both text.

**Model Specifications:**
- 📊 **Parameters**: 220M
- 🏗️ **Architecture**: Encoder-Decoder (12 layers each)
- 🎓 **Pre-training**: C4 corpus with span corruption
- 🔧 **Approach**: Unified text-to-text framework

---

## 📊 Dataset Preparation

### Dataset Source

The **SQuAD (Stanford Question Answering Dataset) v1.1** is used. It consists of questions posed by crowdworkers on a set of Wikipedia articles, where the answer to every question is a segment of text.

**Dataset Statistics:**
- 📚 Training samples: ~87,599
- ✅ Validation samples: ~10,570
- 📖 Source: Wikipedia articles
- 🎯 Task: Extractive QA converted to generative

### Text-to-Text Formatting

To adapt SQuAD for T5, the data is preprocessed into a unified text prompt format:

**Input Format:**
```
question: [Question Text] context: [Paragraph Text]
```

**Target Format:**
```
[Answer Text]
```

**Example:**
```python
# Input
"question: When did the Norman language develop? context: The Normans (Norman: Nourmands; French: Normands; Latin: Normanni) were the people who in the 10th and 11th centuries gave their name to Normandy..."

# Target
"10th and 11th centuries"
```

This format explicitly guides the model to attend to the context to find the answer to the specific question.

---

## 🧠 Tokenization

- **Tokenizer:** `AutoTokenizer` (specifically `T5TokenizerFast`) is used to prevent vocabulary mismatch errors (SentencePiece "piece id out of range")
- **Truncation:** Inputs are padded and truncated to a maximum length of **384 tokens**
- **Target Length:** Answers are truncated to **64 tokens**
- **Dynamic Padding:** Handled via `DataCollatorForSeq2Seq`

```python
def preprocess_function(examples):
    inputs = [f"question: {q} context: {c}" 
              for q, c in zip(examples['question'], examples['context'])]
    
    model_inputs = tokenizer(
        inputs,
        max_length=384,
        truncation=True,
        padding=False
    )
    
    labels = tokenizer(
        examples['answers']['text'],
        max_length=64,
        truncation=True,
        padding=False
    )
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs
```

---

## ⚙️ Model Configuration

The project is optimized to run on **Google Colab Free Tier (T4 GPU)**.

| Parameter | Value |
|-----------|-------|
| **Architecture** | `t5-base` (220M Parameters) |
| **Precision** | Mixed Precision (FP16) |
| **Gradient Accumulation** | 4 steps |
| **Effective Batch Size** | 32 |
| **Physical Batch Size** | 8 |
| **Optimizer** | AdamW |

**Memory Optimization:**
- ✅ FP16 training enabled
- ✅ Gradient accumulation for larger effective batch
- ✅ Dynamic padding to reduce wasted computation
- ✅ Gradient checkpointing available if needed

---

## 🏋️ Training Details

| Configuration | Value |
|--------------|-------|
| **Framework** | Hugging Face `Seq2SeqTrainer` |
| **Hardware** | Google Colab (Tesla T4 GPU) |
| **Epochs** | 2 |
| **Learning Rate** | 2e-5 |
| **Batch Size** | 8 per device |
| **Gradient Accumulation** | 4 steps |
| **Warmup Steps** | 500 |
| **Weight Decay** | 0.01 |

**Training Features:**
- ✅ Early stopping to prevent overfitting
- ✅ Automatic mixed precision (AMP)
- ✅ Learning rate scheduling with warmup
- ✅ Model checkpointing
- ✅ Evaluation during training

The training pipeline handles dynamic padding via `DataCollatorForSeq2Seq` to ensure efficient processing.

---

## 🚀 Inference and Usage

After training, the model can generate answers for any custom question and context.

### Basic Inference

```python
def generate_answer(question, context):
    input_text = f"question: {question} context: {context}"
    inputs = tokenizer(
        input_text, 
        return_tensors="pt", 
        max_length=384, 
        truncation=True
    ).to(device)
    
    outputs = model.generate(
        inputs.input_ids, 
        max_length=64, 
        num_beams=4, 
        early_stopping=True
    )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
```

### Example Usage

```python
# Example 1: Historical Question
question = "When did the Norman conquest of England occur?"
context = "The Norman conquest of England was the 11th-century invasion and occupation of England by an army of Norman, Breton, and French soldiers led by Duke William II of Normandy."
answer = generate_answer(question, context)
print(f"Answer: {answer}")
# Output: "11th-century"

# Example 2: Scientific Question
question = "What is the speed of light?"
context = "The speed of light in vacuum, commonly denoted c, is a universal physical constant important in many areas of physics. Its exact value is 299,792,458 metres per second."
answer = generate_answer(question, context)
print(f"Answer: {answer}")
# Output: "299,792,458 metres per second"
```

### Advanced Generation Parameters

```python
outputs = model.generate(
    inputs.input_ids,
    max_length=64,
    num_beams=4,              # Beam search for better quality
    early_stopping=True,       # Stop when all beams finish
    temperature=0.7,           # Control randomness
    top_k=50,                  # Top-k sampling
    top_p=0.95,                # Nucleus sampling
    do_sample=False            # Deterministic for QA
)
```

---

## 📈 Results and Observations

### Quantitative Metrics

| Metric | Score |
|--------|-------|
| **Exact Match (EM)** | ~76.6% |
| **Training Loss** | 0.28 → 0.26 |
| **Validation Loss** | ~0.30 (stabilized) |
| **Training Time** | ~2-3 hours on T4 |

**Performance Analysis:**
- 🎯 Strong exact match performance for a base model
- 📉 Fast convergence within 2 epochs
- ✅ Low overfitting (train/val loss gap minimal)
- ⚡ Efficient training with FP16

### Qualitative Analysis

The model demonstrates strong capability in understanding context:

**Strengths:**
- ✅ Successfully retrieves dates, entities, and factual information
- ✅ Handles various question types (What, When, Where, Who)
- ✅ Works across different domains (History, Science, Biography)
- ✅ Can paraphrase answers naturally

**Example Predictions:**

```python
# Question about dates
Q: "When did Einstein develop his theory?"
Context: "Einstein developed the theory of relativity in 1905..."
Generated: "1905" ✅

# Question about entities
Q: "Who invented the telephone?"
Context: "Alexander Graham Bell is credited with inventing the telephone in 1876..."
Generated: "Alexander Graham Bell" ✅

# Question requiring reasoning
Q: "What causes gravity?"
Context: "Gravity is caused by the curvature of spacetime around massive objects..."
Generated: "curvature of spacetime" ✅
```

**Observations:**
- 📝 As a generative model, it occasionally shortens answers (e.g., answering "10th and 11th" instead of "10th and 11th centuries")
- 💡 This is technically correct but semantically concise
- 🎯 Shows understanding of core information even when paraphrasing

---

## 📁 Repository Structure

```
finetuning-t5-question-answering/
│
├── notebooks/
│   └── t5_squad_finetuning.ipynb      # Main execution notebook
│
├── reports/
│   └── task2_report.md                # Detailed academic report
│
├── README.md                          # This file
└── requirements.txt                   # Dependencies
```

---

## 🛠️ Installation & Setup

### Requirements

- Python 3.10+
- PyTorch
- Hugging Face Transformers
- SentencePiece
- Accelerate

### Quick Start

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/finetuning-t5-question-answering.git
cd finetuning-t5-question-answering
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Run the training notebook:**
```bash
jupyter notebook notebooks/t5_squad_finetuning.ipynb
```

### Requirements File

Create a `requirements.txt` with:
```txt
torch>=2.0.0
transformers>=4.30.0
datasets>=2.12.0
evaluate>=0.4.0
sentencepiece>=0.1.99
accelerate>=0.20.0
numpy>=1.24.0
pandas>=2.0.0
```

### Alternative Installation

```bash
pip install transformers datasets evaluate sentencepiece accelerate
```

---


## 🙏 Acknowledgments

- **Hugging Face** for the Transformers library and Seq2SeqTrainer
- **Google Research** for the T5 architecture
- **Stanford NLP** for the SQuAD dataset
- **Google Colab** for providing free GPU resources

---

## ✅ Conclusion

This project successfully demonstrates that **T5-base** can be fine-tuned for Generative QA on consumer-grade hardware (Colab T4). The model learned to extract and generate accurate answers across various domains (History, Science, Biography) with high accuracy, proving the effectiveness of the text-to-text transfer learning approach.

**Key Takeaways:**
- 🎯 T5's text-to-text framework is highly effective for QA
- ⚡ Achieves ~76.6% Exact Match on SQuAD v1.1
- 💾 Optimized for resource-constrained environments
- 🔧 Requires only 2 epochs for strong performance
- 🚀 Suitable for production deployment with proper optimization

---

<div align="center">
  <strong>⭐ Star this repository if you find it helpful!</strong>
</div>
