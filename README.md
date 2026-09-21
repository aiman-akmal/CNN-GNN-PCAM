# Phase-Designated Reduction in Hybrid CNN-GNN Models for Computational Pathology

## Overview
This repository contains a phase-designated fidelity-preserving reduction framework applied to hybrid CNN-GNN architectures in computational pathology. The pipeline combines image-derived representations with relational learning to model local morphology alongside tissue organization. 

## Model Architecture
The predictive pipeline is built on a common MobileNetV3-GraphSAGE backbone:
* **CNN Feature Encoding:** Employs an ImageNet-pretrained MobileNetV3 encoder to extract localized image content for each detected cell-level entity.
* **Graph Learning:** Uses two sequential SAGEConv layers to process the resulting graph through neighborhood aggregation.
* **Classification:** Aggregates graph-level representations via global mean pooling and passes them through fully connected layers to generate binary classification logits.

## Phase-Designated Reduction Framework
To manage the computational demands of multi-stage graph pipelines, this project implements three distinct reduction interventions:
* **Feature-Level Reduction (FLR):** Acts on CNN-derived node features before graph learning by utilizing Incremental Principal Component Analysis (IPCA) to reduce feature dimensionality.
* **Graph-Level Reduction (GLR):** Modifies the graph topology supplied to message passing by replacing the standard spatial Delaunay topology with a semantic feature-space K-Nearest Neighbors (KNN) topology.
* **Embedding-Level Reduction (ELR):** Reduces the number of learned node embeddings retained for graph-level aggregation by applying Top-K pooling.

## Dataset
This project is evaluated using the PatchCamelyon (P-CAM) dataset, which consists of 327,680 RGB hematoxylin and eosin (H&E)-stained image patches extracted from lymph-node whole-slide images. 
* All image patches have dimensions of 96x96 pixels.
* A patch is labelled as benign (class 0) or malignant (class 1).

## Evaluation Configurations
The framework features eight distinct configurations to evaluate reduction mechanisms both individually and in combination. These configurations are assessed across predictive correctness (classification accuracy), PGExplainer-guided perturbation fidelity, and training-time computational burden.

## Project Structure
* `src/`: Core Python library.
* `notebooks/`: Execution workflows
  * `01_eda.ipynb`: Exploratory data analysis and patch inspection.
  * `02_feature_extraction.ipynb`: IPCA fitting and dynamic cell crop extraction.
  * `03_visualization.ipynb`: Visual mapping of semantic KNN graph topology.
  * `04_training.ipynb`: PyTorch Geometric training loops.
  * `05_ablation.ipynb`: Configuration evaluations (M1-M8) and PGExplainer fidelity extraction.
  * `06_results.ipynb`: Multi-model ROC and Reliability (Calibration) Curve generation.
* `models/`: Directory dedicated to storing trained PyTorch `.pth` weights.
* `outputs/` & `results/`: Directories for generated visual artifacts and intermediate JSON evaluation metrics.

## Data Download 
Clone or download the official P-CAM dataset from its original repository: [PatchCamelyon (PCam)](https://github.com/basveeling/pcam.git)

## Setup and Usage

1. Clone this repository to your local machine. 

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```