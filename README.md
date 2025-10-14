# MR-Pruner: Multi-Resolution Token Pruning Method

## 📌 Introduction
We propose **MR-Pruner**, a **training-free, graph-based token pruning framework** explicitly designed for **multi-resolution MLLMs**. Unlike prior single-resolution pruning methods, MR-Pruner accounts for the **informativeness distribution across resolutions** and their **mutual complementarity**.  

Key highlights:
- 🔍 **Resolution-aware pruning**: Separately handles high- and low-resolution tokens with adaptive ratios.  
- 🔗 **Graph-based scoring**: Builds intra- and cross-resolution token graphs to propagate informativeness.  
- 🚀 **Training-free**: Plug-and-play, no model retraining needed.  
- 🎯 **Robust**: Preserves complementary tokens across resolutions, enabling resilience in extreme pruning.  

---

## ⚙️ Installation
MR-Pruner is built on [Lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval), which is evaluation toolkit for MLLMs.  

```bash
git clone https://github.com/gooriiie/MR-Pruner.git
cd MR-Pruner

# create environment
conda create -n mrpruner python=3.10
conda activate mrpruner
pip install -r requirements.txt

# create environment with yaml
conda env create -f environment.yml
```

---

## 🚀 Usage

Run evaluation with default settings.
```bash
# TextVQA
bash ./scripts/eval_lmms_eval_textvqa.sh  

# ScienceQA-IMG
bash ./scripts/eval_lmms_eval_sqa.sh

# POPE
bash ./scripts/eval_lmms_eval_pope.sh  
```
