# Experiment Execution

Experiments were conducted on the **VLAB UFSC cluster** using **NVIDIA HGX H100 GPUs**.
Due to hardware requirements, training was executed through Jupyter environments.

From the project root directory, experiments can be reproduced following the steps below.

---

# 1. Download Dataset

Download the dataset from Zenodo:

https://zenodo.org/records/17743609

After downloading, place the dataset inside:

```
data/standard/
```

---

# 2. Prepare Dataset

After the download, apply the dataset preparation scripts.

### Vision Transformer dataset preparation

```
python src/data_utils/prepare_vit_dataset.py
```

### YOLO dataset preparation

```
python src/data_utils/prepare_yolo_dataset.py
```

These scripts merge the original dataset splits (`train`, `valid`, `test`) into a unified dataset used by the training pipelines.

---

# 3. Train Models

Model training scripts are located in:

```
src/models/
```

### Vision Transformer

Standard training:

```
python src/models/vit/train_vit_standard.py
```

Training with HistAuGAN:

```
python src/models/vit/train_vit_histaugan.py
```

### YOLOv11

Standard training:

```
python src/models/yolo/train_yolo_standard.py
```

Training with HistAuGAN:

```
python src/models/yolo/train_yolo_histaugan.py
```

---

# 4. Evaluation

## Linear Probing

```
python src/evaluation/linear_probing/linear_probing_pbc_5fold.py
python src/evaluation/linear_probing/linear_probing_lisc_5fold.py
```

## Full Fine-Tuning

```
python src/evaluation/full_finetuning/full_finetuning_pbc.py
python src/evaluation/full_finetuning/full_finetuning_lisc.py
```

## Zero-Shot Evaluation

```
python src/evaluation/zero_shot/zero_shot_head_pbc.py
python src/evaluation/zero_shot/zero_shot_head_lisc.py
```

---

# 5. Model Explainability

Attention visualization scripts:

```
python src/explainability/vit_attention_rollout.py
python src/explainability/vit_rollout_plot.py
```
