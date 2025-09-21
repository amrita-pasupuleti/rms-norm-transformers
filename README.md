# RMSNorm Transformers: Precision and Quantization Experiments

This repository investigates how **Root Mean Square Layer Normalization (RMSNorm)** performs under different numeric precisions (FP32, FP16) and simplified quantization (INT8). The experiments cover:

- **DistilBERT** models with LayerNorm replaced by RMSNorm on **DBPEDIA-14**.
- A **custom Transformer classifier** for **AG News** with FP32, FP16 (mixed precision), and a toy INT8 RMSNorm.

The goal is to measure **accuracy, training time, inference time, and GPU memory usage** across these configurations.


---

## Repository Structure

```
amrita-pasupuleti-rms-norm-transformers/
├── README.md
├── results_fp16.json           # DistilBERT RMSNorm FP16 results
├── results_fp32.json           # DistilBERT RMSNorm FP32 results
├── results_int8.json           # DistilBERT RMSNorm toy INT8 results
├── rmsnorm_fp32.ipynb          # Notebook: DistilBERT RMSNorm FP32
├── rmsnorm_int8.ipynb          # Notebook: DistilBERT RMSNorm toy INT8
└── custom_transformer/
    ├── 32bit_rms_norm.ipynb    # Custom Transformer with FP32 RMSNorm
    ├── 16bit_rms_norm.ipynb    # Custom Transformer with FP16 RMSNorm
    ├── 8bit_rms_norm.ipynb     # Custom Transformer with toy INT8 RMSNorm
    └── classifier_transformer.ipynb # Baseline Transformer encoder
```

---

## Environment Setup

Requirements:

- Python 3.10+
- PyTorch (with CUDA if available)
- Hugging Face `transformers`, `datasets`
- `accelerate`
- Jupyter Notebook

Install dependencies:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers datasets accelerate jupyter
```

---

## Running the Experiments

### DistilBERT with RMSNorm (DBPEDIA-14)

Use:
- `rmsnorm_fp32.ipynb` for FP32 baseline
- `rmsnorm_int8.ipynb` for toy INT8 RMSNorm

These notebooks:
- Replace all `LayerNorm` layers with `RMSNorm`.
- Train on a DBPEDIA-14 subset (5k training, 1k testing).
- Log metrics into `results_fp32.json`, `results_fp16.json`, and `results_int8.json`.

### Custom Transformer with RMSNorm (AG News)

Use:
- `custom_transformer/32bit_rms_norm.ipynb` for FP32
- `custom_transformer/16bit_rms_norm.ipynb` for FP16
- `custom_transformer/8bit_rms_norm.ipynb` for toy INT8
- `custom_transformer/classifier_transformer.ipynb` for baseline

These notebooks:
- Build a small Transformer with RMSNorm in encoder blocks.
- Use a simple regex-based tokenizer and vocabulary.
- Compare FP32, FP16, and toy INT8 accuracy and efficiency.

---

## Results Summary

### DistilBERT on DBPEDIA-14

| Precision | Test Accuracy | Train Time (s) | Test Time (s) | GPU Memory (MB) |
|-----------|---------------|----------------|---------------|-----------------|
| FP32      | 0.969         | 179.0          | 0.89          | 2686            |
| FP16      | 0.963         | 211.5          | 1.19          | 1461            |
| INT8 (toy)| 0.391         | 165.6          | 0.98          | 1588            |

**Findings**:
- FP16 achieves nearly identical accuracy to FP32 with **~45% less GPU memory**.
- The toy INT8 RMSNorm degrades significantly due to lack of quantization-aware training:contentReference[oaicite:1]{index=1}:contentReference[oaicite:2]{index=2}.

### Custom Transformer on AG News

- FP32 RMSNorm: ~0.91 accuracy
- FP16 RMSNorm: ~0.90–0.91 accuracy, reduced memory/time
- Toy INT8 RMSNorm: ~0.42–0.48 accuracy

---

## Implementation Notes

- RMSNorm implemented as a drop-in replacement for LayerNorm.
- Toy INT8 RMSNorm applies naive fake quantization (rounding to 8-bit scale).
- For meaningful INT8 results, use **Quantization-Aware Training (QAT)** or **Post-Training Quantization (PTQ)** with calibration:contentReference[oaicite:3]{index=3}.

---

## Next Steps

- Explore **FP8** with hardware support (e.g., NVIDIA H100).
- Implement **RMSNorm-aware quantization** strategies.
- Compare against **LayerNorm baselines** in both DistilBERT and custom models.
- Scale experiments to larger datasets and models.

---


## Acknowledgments

- Hugging Face `transformers` and `datasets`
- PyTorch `accelerate`
- DBPEDIA-14 and AG News datasets
