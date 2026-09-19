# Exploring CNN Architectures for Fashion-MNIST

A deep-learning experiment that evaluates convolutional neural-network (CNN) design choices for classifying Fashion-MNIST images. The analysis uses a controlled, iterative workflow to compare validation performance against model complexity.

## Overview

Fashion-MNIST is a dataset of 70,000 greyscale images of clothing items across 10 classes. This project investigates how CNN architecture and training choices affect classification performance.

Rather than selecting a model immediately, the notebook changes one design choice at a time, compares validation-accuracy learning curves, and evaluates the selected architecture on the held-out test set.

## Notebook

[`exploring_cnn_architectures_for_fashion_mnist.ipynb`](exploring_cnn_architectures_for_fashion_mnist.ipynb) contains the complete analysis, including explanatory notes, model definitions, experiments, visualisations, and results.

## Method

The notebook is implemented in Python with TensorFlow/Keras. The workflow is:

1. Load the Fashion-MNIST training and test data and rescale pixel values to the range from 0 to 1.
2. Create a validation split from the training data.
3. Train CNN variants using Adam optimisation, categorical cross-entropy loss, a fixed random seed, and a learning-rate schedule.
4. Compare validation-accuracy curves across experiments.
5. Evaluate the selected model on the held-out test set.

The experiments examine:

- the number of convolution–pooling blocks;
- the number of convolutional feature maps;
- dense-layer size;
- dropout rate;
- pairs of 3-by-3 convolutions compared with 5-by-5 convolutions;
- strided convolutions compared with max pooling;
- batch normalisation; and
- data augmentation.

## Results

The experiments identify a compact model with two convolutional blocks as an effective balance between validation performance and computational complexity. The selected architecture uses paired 3-by-3 convolutions, max pooling, batch normalisation, 20% dropout, and a 128-unit dense layer:

```text
Input → [16C3 → 16C3 → max pool] → [32C3 → 32C3 → max pool] → 128-unit dense layer → 10-class output
```

## Requirements

Install the required Python packages with:

```bash
pip install -r requirements.txt
```

The project uses TensorFlow/Keras, NumPy, Pandas, Matplotlib, scikit-learn, and Jupyter.

## Running the notebook

Open the notebook in Jupyter, JupyterLab, Kaggle, or Google Colab:

```bash
jupyter notebook exploring_cnn_architectures_for_fashion_mnist.ipynb
```

The full experiment suite is computationally intensive, so a GPU-enabled environment such as Kaggle or Google Colab is recommended.

## Data

This project uses Fashion-MNIST. The notebook was originally configured to access the dataset through Kaggle paths; the raw data files are not included in this repository.

To run the analysis outside Kaggle, download the Fashion-MNIST CSV files from the original dataset source and update the `pd.read_csv(...)` paths in the notebook to point to their location on your machine.

Please consult the original dataset source for documentation, citation requirements, licence information, and terms of use.

## Repository contents

- `exploring_cnn_architectures_for_fashion_mnist.ipynb` — complete notebook containing the analysis, experiments, and outputs.
- `requirements.txt` — Python dependencies.
- `.gitignore` — excludes local and generated files from version control.
