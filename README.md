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

