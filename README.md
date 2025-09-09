Holographic Geometric Consistency (HGC) - Thesis Research
This repository contains the complete codebase, experiments, and analysis for the research project "Holographic Geometric Consistency: A Framework for Architecturally Reliable and Economical Language Models."

Project Vision
This research aims to develop and validate the HGC framework, a novel architectural paradigm designed to make Large Language Models intrinsically reliable and economically sustainable. The work is structured into three distinct, publishable phases that collectively prove HGC offers a practical solution to catastrophic forgetting and a multi-layered defense against hallucinations.

Project Structure
HGC_Thesis/
├── .gitignore         # Specifies files for Git to ignore
├── README.md          # This file
├── requirements.txt   # Project dependencies
├── hgc_core/          # Core Python module for the HGC architecture
├── notebooks/         # Jupyter notebooks for each experimental step
├── data/              # (Ignored by Git) Raw and processed datasets
├── results/           # (Ignored by Git) Saved models, figures, and metrics
└── thesis/            # (Future) Markdown/LaTeX files for the thesis chapters

Setup & Reproduction
To replicate this research environment, follow these steps:

Clone the repository:

git clone [https://github.com/JustinArndtAI/HGC_Thesis.git](https://github.com/JustinArndtAI/HGC_Thesis.git)
cd HGC_Thesis

Create and activate a virtual environment:

python -m venv venv
.\venv\Scripts\activate

Install dependencies:

pip install -r requirements.txt

Install PyTorch with CUDA support by following the instructions on the official PyTorch website.

Research Phases
This project is divided into three core phases, with each intended to result in a standalone ArXiv pre-print:

Phase 1: Proving the economic viability of HGC by solving catastrophic forgetting in bert-base-uncased.

Phase 2: Quantifying the architectural reliability of HGC via a full ablation study on gpt2-large.

Phase 3: A deep theoretical dive into the QIEP mechanism and the final synthesis of all findings.

Status
Current Status: Work in progress.
Active Phase: Phase 1 - Data Preparation & Model Building.