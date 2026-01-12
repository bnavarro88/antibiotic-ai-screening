# 🔬 Hybrid AI-Model for Antibiotic Susceptibility Prediction

This repository contains a deep learning framework designed to predict the efficacy of small molecules against specific bacterial phenotypes. The project bridges the gap between **chemoinformatics** and **microbiology** by integrating molecular fingerprints with bacterial cell wall characteristics.

## 🚀 Live Demo
You can test the model with any SMILES string here: 
[INSERT_YOUR_STREAMLIT_URL_HERE]

## 🧠 Methodology

### 1. Feature Engineering
- **Chemical Input:** 1024-bit Morgan Fingerprints (Radius 2) generated via RDKit to capture molecular substructures.
- **Biological Context:** A binary feature vector representing Gram-stain physiology (Positive/Negative), allowing the model to "understand" the target's physical barriers.

### 2. Architecture
A **Hybrid Multi-Layer Perceptron (MLP)** that concatenates chemical and biological embeddings. This design prevents the model from being "bacteria-blind" and forces it to learn the synergy between drug structure and cell wall architecture.

### 3. Key Findings & Validation
- **Model Generalization:** The system was validated using out-of-distribution compounds. 
- **The Daptomycin Case:** Despite not being in the training set, the model correctly identified **Daptomycin** as a highly selective agent for Gram-positive bacteria with **86.6% confidence**, effectively predicting its inability to cross the double membrane of Gram-negatives.

## 🛠️ Tech Stack
- **Deep Learning:** PyTorch
- **Chemoinformatics:** RDKit
- **Interface:** Streamlit
- **Data Source:** Integration of PubChem API and genomic metadata.

---
*Developed as an independent research project to explore AI-driven solutions for the antibiotic resistance crisis.*
