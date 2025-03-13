# ECG Foundation Model with Masked Autoencoder (MAE)

A PyTorch implementation for pretraining and finetuning an ECG analysis model using masked autoencoding and transformer architectures.

## Key Features
- **MAE-based ECG Representation Learning**: Self-supervised pretraining via patch masking/shuffling
- **Flexible Classification**: Fine-tuning with either transformer-based (`ViT_Classifier`) or MLP heads
- **HPC-Ready**: SLURM job scripts for distributed training on GPU clusters
- **Modular Design**: Separated components for model architecture, data handling, and training logic

## Project Structure

### Core Modules
| File | Description |
|------|-------------|
| **`support_model.py`** | Implements MAE architecture components:<br>- `PatchShuffle`: Patch masking/shuffling logic<br>- `MAE_Encoder`/`MAE_Decoder`: Core autoencoder<br>- `ViT_Classifier`/`MLP_Classifier`: Downstream task heads |
| **`support_dataset.py`** | ECG data handling:<br>- `ECG_dataset()`: Data loader with normalization<br>- `CustomTensorDataset`: PyTorch Dataset wrapper<br>- `read_ECG()`: Raw data loading & preprocessing |
| **`support_based.py`** | Utilities for:<br>- Experiment tracking (UUID generation)<br>- Metrics calculation (AUC, accuracy)<br>- Results serialization |
| **`support_args.py`** | Unified command-line arguments for:<br>- Pretraining & finetuning configurations<br>- Model hyperparameters<br>- Dataset paths |

### Training & Evaluation
| File | Purpose |
|------|---------|
| **`model_run.py`** | Main training loop for both pretraining and finetuning |
| **`model_optimization.py`** | Contains train/eval epoch logic and loss calculations |
| **`main_pretrain.py`** | Launch script for pretraining |
| **`main_finetune.py`** | Launch script for fine-tuning |

### Analysis & Results Processing
| File | Function |
|------|----------|
| **`AA99_01_test.py`** | Load trained models and calculate performance metrics |
| **`AA01_01_read_results.py`** | Aggregate results from multiple experimental runs |

### HPC Job Submission
| Script | Purpose |
|--------|---------|
| **`main_pretrain.sbatch`** | SLURM job script for pretraining on GPU clusters |
| **`main_finetune.sbatch`** | SLURM job script for fine-tuning |
