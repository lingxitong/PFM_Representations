# 🧪 PFM_Representations

面向用户的病理 ROI 特征提取与评测工具。

**语言**: [English](./README.md) | 中文

---

## 🎯 这个项目能做什么

PFM_Representations 可以帮助你：
- 将 ROI 图像提取为 `.pt` 特征文件
- 在相同数据集上对比基础模型
- 执行 Linear Probe、KNN、Proto、Few-shot、Zero-shot 等评测

当前仓库仅保留 **4 个模型**：
- `conch_v1`
- `uni_v2`
- `virchow2`
- `hoptimus1`

---

## 🚀 快速开始

### 🧩 1）配置模型权重

编辑 `model_utils/model_weights.json`：

```json
{
  "conch_v1": "",
  "uni_v2": "",
  "virchow2": "",
  "hoptimus1": ""
}
```

- 配置为空字符串 `""`：尝试从 Hugging Face 自动下载
- 配置为本地路径：优先加载本地权重

### 🛠️ 2）提取特征

运行 `00-ROI_Feature_Extract.py`。

示例：

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

### 📊 3）运行评测

使用提取出的特征文件运行 `01-ROI_BenchMark_Main.py`：

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

## ⚙️ 新参数重点说明（特征提取）

这些参数用于控制 **ViT 内部哪个位置的特征** 被导出。

- `--feature_layer {ln2,fc1,act,rc2}`
  - `ln2`：block 内第二个 LayerNorm 后的输出
  - `fc1`：MLP 第一层线性层输出
  - `act`：MLP 激活后的输出
  - `rc2`：完整 block 输出（第二个残差后）

- `--block_index INT`
  - 单层提取（支持负索引，如 `-1`）

- `--block_indices INT [INT ...]`
  - 多层提取

- `--fusion {mean,concat}`
  - 多层特征融合方式
  - `mean`：逐层求平均
  - `concat`：逐层拼接

- `--token_pool {cls,mean,cls_mean}`
  - `cls`：取 CLS token
  - `mean`：patch token 均值
  - `cls_mean`：CLS + patch 均值拼接

### 🔗 参数耦合关系（重要）

- `block_index` 与 `block_indices` 互斥，只能二选一。
- 两者都不传时，默认使用最后一个 block。
- `fusion` 仅在多层提取时生效。

---

## 🏷️ 输出文件命名规则

特征文件会自动带上关键超参数：

```text
Dataset_[{dataset}]_Model_[{model}]_HP_[FL-...__BI-...__BIS-...__FU-...__TP-...__RS-...__BS-...]_{train|test}.pt
```

便于你从文件名直接追溯提取配置。

---

## 📚 常见用户用法

### 🅰️ A）单层提取

```bash
--feature_layer ln2 --block_index 12 --fusion mean --token_pool cls
```

### 🅱️ B）多层融合

```bash
--feature_layer fc1 --block_indices 6 12 18 23 --fusion mean --token_pool mean
```

### 🆑 C）高维特征（维度更大）

```bash
--feature_layer act --block_indices 8 16 24 31 --fusion concat --token_pool cls_mean
```

---

## 📦 脚本说明

- `00-ROI_Feature_Extract.py`：特征提取
- `01-ROI_BenchMark_Main.py`：任务评测
- `02-Bootstrap_Statistical_Analysis.py`：Bootstrap 置信区间分析

---

## 📝 注意事项

- `hoptimus1` 依赖 `timm==0.9.16`。
- 部分 Hugging Face 模型可能是 gated，需要先申请权限。
- `fusion=concat` 会显著增加特征维度和存储/显存开销。

如果需要，我可以继续补一份“自动扫参脚本”，直接批量遍历这些新参数。
