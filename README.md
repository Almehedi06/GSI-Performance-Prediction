# GSI-Performance-Prediction

Welcome to my repository on **AI-based modeling for Green Stormwater Infrastructure (GSI) performance prediction**.  
This project implements a **scalable, configurable ML/DL pipeline** for evaluating and comparing multiple models under a unified structure.

## Why This Matters
Green Stormwater Infrastructure (GSI) plays a key role in **reducing urban flooding, improving water quality**, and **supporting urban ecosystems**.  
However, traditional models are **computationally expensive** and **hard to generalize**.  
This project aims to **overcome these limitations** by providing a **flexible ML/DL framework** with **model configurability** and **reproducibility**.

## Supported Models
This repository supports **training and testing** of the following models:

✅ **Random Forest (RF)** — Non-parametric ensemble learning.  
✅ **Data-Driven LSTM (LSTM)** — Sequence modeling for time-series data.  
✅ **Physics-Informed LSTM (PILSTM)** — Deep learning with physical constraints (coming soon).

Model selection is **fully controlled via `config/config.yaml`** or **CLI arguments**.

## Project Structure
<pre> ```text ├── config/ # Model and data configuration (YAML) ├── data/ # Raw and processed data (DO NOT COMMIT large files) ├── notebooks/ # Experimentation notebooks ├── results/ # Logs, models, metrics, and plots ├── src/ # Source code for models, utils, and pipelines │ ├── models/ # Model definitions (RF, LSTM, PILSTM) │ ├── utils/ # Utility functions for data and evaluation │ ├── train.py # Model training pipeline │ ├── test_model.py # Model testing pipeline └── README.md # Project overview (this file) ``` </pre>

shell
Copy
Edit

## Usage Example

### Train Random Forest:
```bash
python -m src.train --model rf
Train LSTM:
bash
Copy
Edit
python -m src.train --model lstm
Test Random Forest:
bash
Copy
Edit
python -m src.test_model --model rf
Test LSTM:
bash
Copy
Edit
python -m src.test_model --model lstm
Key Contributions
📌 Unified, scalable ML/DL pipeline with CLI and YAML config support.
📌 Supports Random Forest, LSTM, and Physics-Informed LSTM.
📌 Reproducible evaluation on urban stormwater ponding depth (dPD).
📌 Extensible structure for adding future models or workflows.

How to Get Started
Update config/config.yaml with your data and model settings.

Use the provided CLI commands to train or test models.

Check results/ for saved models and evaluation metrics.

🚀 Feel free to fork, explore, and contribute to advance AI-powered urban water management! 🚀

yaml
Copy
Edit

---

Let me know if you'd like me to draft a **CONTRIBUTING.md** or **LICENSE.md** next.
