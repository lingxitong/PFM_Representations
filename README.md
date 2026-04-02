# HistoROIBench

A user-friendly toolkit for extracting ROI features from pathology images and running downstream evaluation.

**Language**: English | [中文说明](./README_CN.md)

---

## What This Project Is For

HistoROIBench helps you:
- extract ROI features into `.pt` files
- compare foundation models on the same dataset
- run downstream tasks such as Linear Probe, KNN, Proto, Few-shot, and Zero-shot

This repository is currently streamlined to **4 models**:
- `conch_v1`
- `uni_v2`
- `virchow2`
- `hoptimus1`

---

## Quick Start

### 1) Prepare model weights

Edit `model_utils/model_weights.json`:

```json
{
  "conch_v1": "",
  "uni_v2": "",
  "virchow2": "",
  "hoptimus1": ""
}
```

- Empty string (`""`) means auto-download from Hugging Face (if accessible).
- A local path means loading from your local checkpoint.

### 2) Extract features

Run `00-ROI_Feature_Extract.py`.

Example:

```bash
python 00-ROI_Feature_Extract.py \
  --dataset_split_csv ./example_dataset/CRC-100K.csv \
  --class2id_txt ./example_dataset/CRC-100K.txt \
  --dataset_name CRC-100K \
  --model_name virchow2 \
  --feature_layer ln2 \
  --block_index 20 \
  --fusion mean \
  --token_pool cls_mean \
  --batch_size 128 \
  --device cuda:0 \
  --save_dir ./ROI_Features
```

### 3) Run benchmark

Run `01-ROI_BenchMark_Main.py` with extracted feature files:

```bash
python 01-ROI_BenchMark_Main.py \
  --TASK Linear-Probe,KNN,Proto \
  --train_feature_file ./ROI_Features/your_train.pt \
  --test_feature_file ./ROI_Features/your_test.pt \
  --class2id_txt ./example_dataset/CRC-100K.txt \
  --log_dir ./results \
  --device cuda:0
```

---

## Feature Extraction Parameters (New)

These parameters control **which internal ViT representation** is exported.

- `--feature_layer {ln2,fc1,act,rc2}`
  - `ln2`: output after second layer norm in a block
  - `fc1`: output after first MLP linear layer
  - `act`: output after MLP activation
  - `rc2`: output of the full block (after second residual)

- `--block_index INT`
  - extract from one block (supports negative index, e.g. `-1`)

- `--block_indices INT [INT ...]`
  - extract from multiple blocks

- `--fusion {mean,concat}`
  - used when multiple blocks are selected
  - `mean`: average block features
  - `concat`: concatenate block features

- `--token_pool {cls,mean,cls_mean}`
  - `cls`: use CLS token
  - `mean`: average patch tokens
  - `cls_mean`: concat CLS + patch mean

### Important coupling rules

- `block_index` and `block_indices` are mutually exclusive.
- If neither is set, the last block is used.
- `fusion` only matters for multi-block extraction.

---

## Output File Naming

Output filenames include key hyperparameters automatically:

```text
Dataset_[{dataset}]_Model_[{model}]_HP_[FL-...__BI-...__BIS-...__FU-...__TP-...__RS-...__BS-...]_{train|test}.pt
```

This makes it easy to trace each feature file back to its extraction setup.

---

## Typical User Recipes

### A) Single-layer extraction

```bash
--feature_layer ln2 --block_index 12 --fusion mean --token_pool cls
```

### B) Multi-layer fusion

```bash
--feature_layer fc1 --block_indices 6 12 18 23 --fusion mean --token_pool mean
```

### C) High-capacity feature (larger dimension)

```bash
--feature_layer act --block_indices 8 16 24 31 --fusion concat --token_pool cls_mean
```

---

## Included Scripts

- `00-ROI_Feature_Extract.py`: feature extraction
- `01-ROI_BenchMark_Main.py`: benchmark tasks
- `02-Bootstrap_Statistical_Analysis.py`: statistical confidence analysis

---

## Notes

- `hoptimus1` requires `timm==0.9.16`.
- Some models are gated on Hugging Face and require access permission.
- `fusion=concat` increases feature dimensionality and memory/storage cost.

If you want, you can add a small automation script to sweep combinations of `feature_layer`, `block_index/block_indices`, `fusion`, and `token_pool`.
