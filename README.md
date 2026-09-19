# Phase-Designated Reduction in Hybrid CNN-GNN Models for Computational Pathology

## Overview
This repository contains phase-designated fidelity-preserving reduction framework applied to hybrid CNN-GNN architectures in computational pathology. The pipeline combines image-derived representations with relational learning to model local morphology alongside tissue organization. 

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
* A patch labelled as benign (class 0) or malignant (class 1)

## Evaluation Configurations
The framework features eight distinct configurations to evaluate reduction mechanisms both individually and in combination. These configurations are assessed across predictive correctness (classification accuracy), PGExplainer-guided perturbation fidelity, and training-time computational burden.

## Project Structure
* `data/`: Directory for storing the P-CAM dataset and preprocessed HDF5 files.
* `app.py`: The Streamlit interface for end-user interaction.
* `src/`: Core Python modules containing the CNN-GNN model architecture, dataset loaders, and inference logic.
* `scripts/`: Executable Python scripts for end-to-end pipeline execution
* `models/`: Directory dedicated to storing trained PyTorch `.pth` weights.
* `outputs/`: Directory where generated visualization artifacts are saved.


## Setup and Installation

1. Clone this repository to your local machine:
   ```bash
   git clone [https://github.com/aiman-akmal/CNN-GNN-PCAM.git](https://github.com/aiman-akmal/CNN-GNN-PCAM.git)
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Launch the Streamlit application:
   ```bash
   streamlit run app.py
   ```