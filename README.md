# DEEPALI-IMAGE-REGISTRATION

## Introduction

Image registration is a fundamental task in medical image analysis, involving the spatial alignment of two or more images. It's crucial for various applications, including disease progression monitoring, multi-modal analysis, and population studies. This project implements an unsupervised image registration network inspired by VoxelMorph, using the DeepALI framework.

The main goals of this project are:
1. To provide a flexible and efficient implementation of unsupervised image registration for medical imaging.
2. To compare the performance of our DeepALI-based implementation with the original VoxelMorph implementation.

This repository contains two main training scripts:
1. `train_mri.py`: Our implementation using the DeepALI framework.
2. `train_voxelmorph.py`: The original VoxelMorph implementation in TensorFlow.

Both scripts use the `config.ini` file for parameter configuration, allowing easy experimentation and comparison between the two implementations.

## Table of Contents
1. [Project Structure](#project-structure)
2. [Requirements](#requirements)
3. [Environment Setup](#environment-setup)
   - [Python Virtual Environment](#python-virtual-environment)
   - [Conda Environment](#conda-environment)
4. [Installation](#installation)
5. [Usage](#usage)
   - [DeepALI Implementation](#deepali-implementation)
   - [VoxelMorph Implementation](#voxelmorph-implementation)
6. [Configuration](#configuration)
7. [License](#license)

## Project Structure

```
DEEPALI-IMAGE-REGISTRATION
├── data
│   └── ixi_t2_dataset_cropped.npy
├── src
│   ├── classes
│   │   ├── dataset.py
│   │   ├── losses.py
│   │   ├── metrics.py
│   │   └── model.py
│   └── scripts
│       ├── data_generation_brats.py
│       ├── data_script_new.py
│       ├── data_script.py
│       ├── train_mri.py
│       └── train_voxelmorph.py
├── .gitignore
├── config.ini
├── LICENSE
├── README.md
├── requirements_tf.txt
└── requirements.txt
```

## Requirements

- For the DeepALI implementation: See `requirements.txt`
- For the original VoxelMorph implementation: See `requirements_tf.txt`

## Environment Setup

### Python Virtual Environment

1. Create a new virtual environment:
   ```
   python -m venv deepali_env
   ```

2. Activate the virtual environment:
   - On Windows:
     ```
     deepali_env\Scripts\activate
     ```
   - On macOS and Linux:
     ```
     source deepali_env/bin/activate
     ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

### Conda Environment

1. Create a new Conda environment:
   ```
   conda create --name deepali_env python=3.8
   ```

2. Activate the Conda environment:
   ```
   conda activate deepali_env
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Installation

Clone the repository and navigate to the project directory:

```
git clone https://github.com/your-username/DEEPALI-IMAGE-REGISTRATION.git
cd DEEPALI-IMAGE-REGISTRATION
```

## Usage

### DeepALI Implementation

To run the DeepALI-based implementation:

```
python src/scripts/train_mri.py
```

This script uses the DeepALI framework to train an unsupervised image registration network.

### VoxelMorph Implementation

To run the original VoxelMorph implementation:

1. Create a separate environment (optional but recommended):
   ```
   python -m venv voxelmorph_env
   source voxelmorph_env/bin/activate  # On Windows, use: voxelmorph_env\Scripts\activate
   ```

2. Install the TensorFlow-specific requirements:
   ```
   pip install -r requirements_tf.txt
   ```

3. Run the VoxelMorph training script:
   ```
   python src/scripts/train_voxelmorph.py
   ```

This script uses the original VoxelMorph architecture using TensorFlow.

## Configuration

Both implementations use a `config.ini` file for parameter configuration. The file is divided into two sections: `[pytorch]` for the DeepALI implementation and `[tensorflow]` for the original VoxelMorph implementation.

### DeepALI Implementation Parameters

- `grid_size_x`: Width of the input images (192)
- `grid_size_y`: Height of the input images (160)
- `batch_size`: Number of samples per batch for training (32)
- `train_val_split`: Ratio of training data to validation data (0.8)
- `val_test_split`: Ratio of validation data to test data (0.5)
- `nb_epochs`: Number of training epochs (5)
- `learning_rate`: Learning rate for the optimizer (0.001)
- `loss`: Loss function used for training (LCC - Local Cross Correlation)
- `loss_weight`: Weight applied to the loss function (0.005)
- `images_path`: Path to the input image dataset (oasis_dataset_1.npy)
- `seg_path`: Path to the segmentation dataset (oasis_dataset_seg.npy)
- `weights_path`: Path to save or load model weights (deepali_vxl/model_weights/weights.h5)

### VoxelMorph Implementation Parameters

- `batch_size`: Number of samples per batch for training (16)
- `train_val_split`: Ratio of training data to validation data (0.8)
- `val_test_split`: Ratio of validation data to test data (0.5)
- `int_steps`: Number of integration steps for the diffeomorphic model (0)
- `lambda_param`: Regularization parameter for the deformation field (0.02)
- `steps_per_epoch`: Number of steps (batches) per epoch (100)
- `nb_epochs`: Number of training epochs (25)
- `verbose`: Verbosity mode (1 - display progress bar)
- `loss`: Loss function used for training (NCC - Normalized Cross Correlation)
- `grad_norm_type`: Type of gradient normalization (l2)
- `gamma_param`: Additional regularization parameter (0.01)
- `learning_rate`: Learning rate for the optimizer (0.001)
- `images_path`: Path to the input image dataset (data/brats_flair_dataset.npy)
- `weights_path`: Path to save or load model weights (vxlmorph/model_weights/weights.h5)
- `patience`: Number of epochs with no improvement after which training will be stopped (60)

To modify these parameters, edit the `config.ini` file in the root directory of the project. The scripts will automatically load these parameters when running.

Note: Ensure that the paths to datasets and weight files are correct and accessible from your working directory.

## License

This project is licensed under the terms of the LICENSE file included in the repository.