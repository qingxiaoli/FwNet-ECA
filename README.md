# FwNet-ECA
Repository for the FwNet-ECA paper, which includes data  and code for the paper.

This repository contains the data and code for the paper *"FwNet-ECA: Communicate different windows through filter enhancement operations and add channel attention"*.

## Structure
- `data/`: Contains a list of links to datasets used in the experiments.
- `code/`: Placeholder for the code, which will be released later.

## Data
The data used for this research can be accessed via the links in the `data/data_links.md` file.

## Code
### Features

- **Custom Model Architecture**: Utilizes the `FWNet_ECA_tiny_patch4_window7_224` model for efficient and accurate image classification.
- **Data Augmentation**: Implements advanced data augmentation techniques including `rand_augment_transform` for robust training.
- **TensorBoard Integration**: Logs training metrics to TensorBoard for easy visualization and monitoring.
- **Flexible Training Configuration**: Supports various command-line arguments to customize training parameters such as learning rate, batch size, and number of epochs.
- **Pre-trained Weights**: Allows loading of pre-trained weights to fine-tune the model or continue training from a checkpoint.
- **Layer Freezing**: Option to freeze certain layers of the model during training.

### Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/FwNet-ECA.git
    cd your-repo
    ```

2. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

### Usage

#### Command-Line Arguments

- `--num_classes`: Number of classes in the dataset (default: 1000).
- `--epochs`: Number of training epochs (default: 100).
- `--batch-size`: Batch size for training (default: 64).
- `--lr`: Learning rate (default: 0.0005).
- `--data-path`: Path to the dataset (default: ``).
- `--weights`: Path to the pre-trained weights file.
- `--weights_dir`: Directory to save the weights.
- `--logdir`: Directory for TensorBoard logs.
- `--freeze-layers`: Whether to freeze layers other than the head (default: False).
- `--if_scheduler`: Whether to use a learning rate scheduler (default: True).
- `--device`: Device to use for training (default: `cuda:0`).

#### Example Command

```sh
python train.py --data-path /path/to/dataset --epochs 50 --batch-size 32 --lr 0.001 --logdir ./logs --weights ./weights/model.pth
