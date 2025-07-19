# Image Captioning with CNN and Transformer

This project implements an image captioning model using a Convolutional Neural Network (CNN) as an image feature extractor and a Transformer-based decoder to generate descriptive captions. The model is trained on the Flickr30k dataset.

## Table of Contents
- [About the Dataset](#about-the-dataset)
- [Model Architecture](#model-architecture)
- [Features](#features)
- [Requirements](#requirements)
- [Usage](#usage)
- [Training Details](#training-details)
- [Evaluation](#evaluation)

## About the Dataset

This project utilizes the **Flickr30k dataset**, a widely-used benchmark for image captioning tasks. The dataset contains 31,783 images sourced from Flickr, each accompanied by five human-generated captions. This results in a total of over 158,000 captions, providing a rich source of data for training models to understand and describe visual scenes.

## Model Architecture

The model follows a standard encoder-decoder architecture:

*   **Encoder:** A pre-trained **EfficientNetB0** is used as the CNN encoder. This model, with its weights frozen, extracts high-level feature representations from the input images.
*   **Decoder:** A **Transformer decoder** is responsible for generating the captions. It consists of a `TransformerDecoderBlock` which includes multi-head self-attention, cross-attention over the image features, and feed-forward neural networks. Positional embeddings are added to the input of the decoder to retain information about the word order in the captions.

## Features

*   **Data Handling:** The script includes robust functions for loading and preprocessing the Flickr30k dataset. This includes parsing the captions, splitting the data into training, validation, and test sets, and creating `tf.data.Dataset` objects for efficient training.
*   **Text and Image Preprocessing:** Text captions are standardized by converting to lowercase and removing punctuation. They are then vectorized using `TextVectorization`. Images are decoded, resized, and undergo data augmentation (random flips, rotations, and contrast adjustments) during training.
*   **Custom Model Components:** The model is built with custom Keras layers, including `TransformerEncoderBlock`, `PositionalEmbedding`, and `TransformerDecoderBlock`, to construct the Transformer architecture.
*   **Training and Evaluation:** The `ImageCaptioningModel` class encapsulates the entire training and evaluation logic, including a custom training step, loss calculation (`SparseCategoricalCrossentropy`), and an accuracy metric.
*   **Inference:** A greedy search algorithm (`greedy_algorithm`) is implemented to generate captions for new images.
*   **Evaluation Metric:** The quality of the generated captions is evaluated using the **BLEU (Bilingual Evaluation Understudy)** score, which compares the generated captions to the ground-truth captions.

## Requirements

*   tensorflow
*   numpy
*   matplotlib
*   nltk
*   scikit-learn
*   tqdm

pip install tensorflow numpy matplotlib nltk scikit-learn tqdm```

## Usage

1.  **Set up the environment:** Make sure you have the required libraries installed. The script is designed to be run in an environment like Kaggle or Google Colab with access to GPU resources for faster training.

2.  **Define Paths:** Update the `IMAGES_PATH` and `CAPTIONS_PATH` variables to point to the location of the Flickr30k image files and the `results.csv` file, respectively.

3.  **Run the script:** Execute the Python script. The script will:
    *   Load and preprocess the data.
    *   Build the image captioning model.
    *   Compile and train the model.
    *   Save the best performing model based on validation loss.
    *   Generate captions for the test set.
    *   Visualize the results with BLEU scores.

4.  **Inference on a new image:** To generate a caption for a new image, you can use the `greedy_algorithm` function by providing the path to your image.

## Training Details

*   **Hyperparameters:**
    *   `SEQ_LEN`: 24 (Maximum sequence length for captions)
    *   `VOCAB_SIZE`: 13000 (Size of the vocabulary)
    *   `IMAGE_SIZE`: (255, 255)
    *   `EMBED_DIM`: 512 (Embedding dimension)
    *   `FF_DIM`: 512 (Feed-forward network dimension)
    *   `BATCH_SIZE`: 256
    *   `EPOCHS`: 100

*   **Optimizer:** The Adam optimizer is used with a custom learning rate schedule (`LRSchedule`) that includes a warm-up phase.

*   **Callbacks:**
    *   `EarlyStopping`: Training is configured to stop early if the validation loss does not improve for 3 consecutive epochs.
    *   `ModelCheckpoint`: The best version of the model is saved to a file based on the minimum validation loss.

## Evaluation

The performance of the model is evaluated using the BLEU score, which measures the similarity between the machine-generated captions and the reference captions. The script calculates BLEU-1, BLEU-2, BLEU-3, and BLEU-4 scores. The `visualization` function displays the generated captions alongside the original images and their corresponding BLEU scores.
