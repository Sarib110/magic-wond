# magic-wand

Random Forest Model Training and Prediction with GUI

## Overview
This project implements a Random Forest-based machine learning pipeline for training, evaluating, and predicting using coordinate data from CSV files. It includes a GUI application for predicting labels of individual files.

## Features
- **Data Loading:** Reads data from CSV files organized in folders where each folder represents a class label
- **Data Preprocessing:** Normalizes and transforms coordinate data into features like relative movements, speed, and angles
- **Feature Extraction:** Calculates statistical features (mean, standard deviation) for predictive modeling
- **Model Training:** Trains a Random Forest classifier on the processed and extracted features
- **Model Saving:** Saves the trained model, label encoder, and scaler for later use
- **Prediction:** Provides a Python script to predict the class label of a single CSV file. Includes a GUI software for user-friendly predictions

## Directory Structure
```
project/
├── All Data/                    # Directory containing labeled CSV files for training
├── full_data_rf_traininig.py
├── gui_predictions.py          # GUI application for predictions
├── rf_all_data_model_files/    # Directory to store the model, scaler, and encoder
├── requirements.txt            # List of dependencies
└── README.md                   # Project documentation
```

## Prerequisites
- Python 3.8 or higher
- Install dependencies using:
  ```
  pip install -r requirements.txt
  ```

## Usage

### Training the Model
1. Place your CSV files inside the All Data directory, organized by class labels (folder names)
2. Run the training script:
   ```
   python train_model.py
   ```

### Prediction
#### Command Line
```
python predict_file.py --file_path path/to/your/file.csv
```

#### GUI Application
1. Launch the GUI:
   ```
   python gui_predictor.py
   ```
2. Select a CSV file using the "Browse" button
3. Click "Predict" to see the class label

## File Descriptions

### train_model.py
Trains a Random Forest classifier using features extracted from coordinate data. Saves the model, label encoder, and scaler for later predictions.

### predict_file.py
Loads the saved model, scaler, and label encoder to predict the label of a single CSV file. Processes and extracts features from the input file before prediction.

### gui_predictor.py
A Tkinter-based graphical user interface for user-friendly predictions. Allows users to select a file and view predictions interactively.

## Dependencies
See requirements.txt for the full list of dependencies:
- numpy
- pandas
- scikit-learn
- opencv-python
- Pillow
- matplotlib
- scipy
- tk

## Model Files
The trained model and associated files are saved in `rf_all_data_model_files`:
- `model.pkl`: Trained Random Forest model
- `label_encoder.pkl`: Label encoder for class labels
- `scaler.pkl`: Scaler for feature normalization

## Future Enhancements
- Add support for additional features or alternative classifiers
- Improve the GUI with advanced functionalities like batch file predictions

## License
This project is licensed under the MIT License.
