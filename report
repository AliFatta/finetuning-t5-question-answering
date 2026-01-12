# Report: Generative Question Answering with T5-Base

## 1. Introduction
Question Answering (QA) is a fundamental task in Natural Language Processing (NLP). Traditional approaches (e.g., BERT) treat QA as a token classification task, identifying the start and end positions of an answer within a context. This project explores a **Generative** approach using the T5 (Text-to-Text Transfer Transformer) model, which treats QA as a Sequence-to-Sequence (Seq2Seq) problem, generating the answer text directly.

## 2. Model Architecture
We utilize **T5-base**, an Encoder-Decoder Transformer pre-trained on a multi-task mixture of unsupervised and supervised tasks.

* **Encoder:** Processes the input sequence (Question + Context) using self-attention mechanisms to create a contextualized representation.
* **Decoder:** Generates the answer token-by-token, attending to the encoder's outputs.
* **Text-to-Text Framework:** T5 casts all NLP tasks into a text-to-text format. For QA, we prepend the prefix "question:" and "context:" to guide the model.

## 3. Experimental Setup
### 3.1 Data Preprocessing
The SQuAD v1.1 dataset was utilized.
* **Input Formatting:** `question: {question} context: {context}`
* **Tokenization:** Used `T5Tokenizer` with a max input length of 384 tokens (to capture sufficient context) and a max target length of 64 tokens.
* **Padding/Truncation:** Applied to ensure uniform batch shapes.

### 3.2 Training Strategy (Colab Optimization)
To ensure execution on Google Colab Free (T4 GPU, 16GB VRAM, 12GB RAM), specific optimizations were applied:
1.  **Batch Size:** Reduced to 8 to prevent Out-Of-Memory (OOM) errors.
2.  **Gradient Accumulation:** set to 4 to simulate a batch size of 32, ensuring stable gradient updates.
3.  **Mixed Precision (FP16):** Enabled to reduce memory footprint and speed up computation.

## 4. Evaluation & Results
The model was evaluated using Exact Match (EM) logic on the generated text.
* **Quantitative:** The model successfully converged, showing decreasing validation loss over 2 epochs.
* **Qualitative:** Inference tests show the model is capable of extracting specific dates and entities accurately from unseen contexts.

## 5. Conclusion
Fine-tuning T5-base for Generative QA is feasible on consumer-grade hardware (Colab Free) using gradient accumulation. The generative approach is more flexible than extractive methods, as it theoretically allows the model to rephrase answers, though in SQuAD it largely learns to copy spans from the text.
