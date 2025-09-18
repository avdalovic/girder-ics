# GIRDER: Graph Neural Network Conformal Prediction for Uncertainty Quantification in Industrial Control Systems

This repository implements graph neural network-based conformal prediction for uncertainty quantification in Industrial Control Systems (ICS). This work is based on the CoRel framework.

## Industrial Control System Datasets

This work focuses on uncertainty quantification for Industrial Control Systems (ICS) using three key datasets:

- **SWaT** (Secure Water Treatment): 25 sensors from a water treatment testbed
- **TEP** (Tennessee Eastman Process): 41 sensors from a chemical process simulation  
- **WADI** (Water Distribution): Sensors from a water distribution testbed

Each dataset is preprocessed with uniform 10-second sampling intervals and proper sensor/actuator separation.

### Dataset Setup

**TEP Dataset**: Included in this repository (`data/TEP/TEP_train.csv`)

**SWaT Dataset**: Requires formal request from https://itrust.sutd.edu.sg/itrust-labs_datasets/dataset_info/ (see `data/SWAT/README.md`)

**WADI Dataset**: Requires formal request from https://itrust.sutd.edu.sg/itrust-labs_datasets/dataset_info/ (see `data/WADI/README.md`)

## Requirements

To solve all dependencies, we recommend using [Anaconda or Miniconda](https://conda.io/projects/conda/en/latest/user-guide/install) to build an environment using the configuration specified on [conda_env.yml](conda_env.yml) by running the command:

```bash
conda env create -f conda_env.yml
```

Once the environment is created, activate it:

```bash
conda activate corel
```

## Quick Start Example

### 1. Train Base Model (TEP Dataset)

```bash
# Train RNN base model on TEP dataset
python -m experiments.run_base_model config=default model=rnn dataset=tep save_outputs=true
```

Results are saved in `logs/base/tep/rnn/YYYY-MM-DD/HH-MM-SS/` - **note this path for the next step**.

### 2. Train Conformal Predictor

```bash
# Train CoRel on TEP with RNN base model (replace timestamp with actual path)
python -m experiments.run_corel config=default model=corel dataset=tep src_dir="./logs/base/tep/rnn/2025-09-15/19-28-51/"
```

## Available Models

- **Base Models**: `rnn`, `transformer`, `stgnn`
- **Conformal Models**: `corel` (graph-based), `cornn` (RNN-based)
- **Baselines**: SCP, SeqCP, NexCP

## Key Features

- **ICS-Optimized**: Preprocessing and configurations specifically tuned for industrial control systems
- **Comprehensive Baselines**: Multiple conformal prediction approaches for comparison
- **Debug Logging**: Extensive debug output to track experiment progress

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
