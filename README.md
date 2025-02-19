# Video Classification Project

This project implements a video classification application using PyTorch. The goal is to classify videos as "REAL" or "FAKE" by extracting frame-level features using a pretrained ResNeXt-50 backbone and modeling temporal dependencies using an LSTM.

## Project Structure

- **Dataset and DataLoader Setup**
  - `video_dataset`: A custom PyTorch Dataset that reads videos from given file paths, extracts frames using OpenCV, and applies image transformations.
  - Data is split into training and validation sets using `train_test_split` from scikit-learn.
  - Transformations for training and testing are applied using `torchvision.transforms`, including resizing, normalization, and conversion to PIL images.

- **Model Architecture**
  - **CNN Backbone**: Uses a ResNeXt-50-32x4d model (without the final classification layers) as a feature extractor.
  - **LSTM**: Processes the sequence of features extracted from the frames.
  - **Classifier**: A fully connected layer with dropout is applied on the averaged LSTM output to generate class predictions.
  - The model is implemented in the `Model` class.

- **Training Utilities**
  - **Training Loop**: The training process is managed by the `train_epoch` function which computes loss and accuracy for each batch.
  - **Validation/Test Loop**: The `test` function evaluates the model on validation data.
  - **Metrics**: Accuracy, precision, recall, F1 score, and AUC are computed using functions from scikit-learn.
  - **Visualization**: Functions `plot_loss` and `plot_accuracy` plot the loss and accuracy curves over epochs. A confusion matrix is plotted using Seaborn.
  - **Early Stopping**: Implemented in the `EarlyStoppingWithCheckpoint` class to monitor validation loss and stop training early if no improvement is seen.

- **Utility Functions**
  - **Number of Videos Calculation**: The helper function `number_of_real_and_fake_videos` counts the number of real and fake videos in a dataset.
  - **Accuracy Calculation**: The `calculate_accuracy` function computes the prediction accuracy.
  - **Checkpointing**: The model checkpoint is automatically saved when there is an improvement in validation loss.

- **Main Execution**
  - The main function handles:
    - Loading video files from `./Models/video_files.pkl`.
    - Reading video metadata from `./Data/Gobal_metadata.csv`.
    - Splitting the dataset into training and validation sets.
    - Initializing the model, optimizer, loss function, and early stopping mechanism.
    - Training the model over a specified number of epochs.
    - Evaluating the trained model and plotting training curves.
    - Displaying the confusion matrix.

## Requirements

- Python 3.7+
- PyTorch
- Torchvision
- OpenCV (cv2)
- NumPy
- Matplotlib
- Pandas
- Seaborn
- Scikit-learn

You can install the required packages using pip:

```bash
pip install torch torchvision opencv-python numpy matplotlib pandas seaborn scikit-learn
```

## Usage

1. **Prepare Your Data:**
   - Ensure all video files are correctly formatted and paths are saved in `./Models/video_files.pkl`.
   - Ensure that the metadata CSV (`./Data/Gobal_metadata.csv`) is available and contains at least two columns: the video file name and its label ("REAL" or "FAKE").

2. **Run the Training Script:**
   - Execute the main script by running:
     ```bash
     python your_script_name.py
     ```
   - When prompted, provide a model name for checkpoint saving.

3. **Monitor Training:**
   - The training progress (loss and accuracy per epoch) will be displayed in the console.
   - Training curves for loss and accuracy will be plotted after training.
   - Confusion matrices and performance metrics will be printed out.

## Customization

- **Model Hyperparameters:**  
  You can adjust various parameters such as sequence length, learning rate, the number of LSTM layers, hidden dimensions, and dropout rate in the `Model` class and training configuration.

- **Transformations:**  
  Customize image transformations in `train_transforms` and `test_transforms` within the `main` function.

- **Early Stopping:**  
  Modify the patience or minimum delta for early stopping in the `EarlyStoppingWithCheckpoint` class as needed.

## Troubleshooting

- **NaN Values:**  
  Warnings are printed if the inputs or outputs contain NaN values during training. Debug these cases by checking your data preprocessing steps.

- **GPU Setup:**  
  Ensure that CUDA is available if running on a GPU. The model and data tensors are moved to the GPU if available.

## License

This project is provided for educational and research purposes. Modify and use as required.

## Acknowledgments

- [PyTorch](https://pytorch.org/)
- [Torchvision](https://pytorch.org/vision/stable/index.html)
- [ResNeXt-50](https://arxiv.org/abs/1611.05431)
- [scikit-learn](https://scikit-learn.org/)
- [OpenCV](https://opencv.org/)