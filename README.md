# Magic-Wand

Random Forest Model Training and Prediction with GUI

## Overview
This project implements a Random Forest-based machine learning pipeline for training and predicting magical gestures using webcam tracking of green objects. The system tracks green objects through a webcam to record gesture patterns, processes the data, and predicts the corresponding magical spell.

## Features
- **Video Tracking:** Real-time tracking of green objects through webcam
- **Data Processing:** Normalizes and transforms coordinate data into features
- **Path Visualization:** Displays the tracked movement path in real-time
- **Data Normalization:** Automatically normalizes and interpolates coordinates
- **Model Training:** Trains a Random Forest classifier on processed features
- **Real-time Prediction:** Provides immediate predictions for recorded gestures
- **Data Export:** Saves recorded gestures as CSV files for further analysis
- **Interactive GUI:** User-friendly interface for recording and visualization

## Directory Structure
```
project/
├── All Data/                    # Directory containing labeled CSV files for training
├── without_cross_validation.py   # Model training script
├── gui_predictor.py            # GUI application for gesture recording and prediction
├── rf_all_data_model_files/    # Directory storing model files
│   ├── model.pkl               # Trained Random Forest model
│   ├── label_encoder.pkl       # Label encoder for spell classes
│   └── scaler.pkl             # Feature scaler
├── requirements.txt            # List of dependencies
└── README.md                   # Project documentation
```

## Prerequisites
- Python 3.8 or higher
- Webcam
- Green object for tracking
- Dependencies (install via requirements.txt):
  ```
  pip install -r requirements.txt
  ```

## Installation
1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Ensure webcam access is available
4. Verify model files are present in rf_all_data_model_files/

## Usage

### Training the Model
1. Organize training data:
   - Place CSV files in All Data directory
   - Each spell type should have its own subdirectory
   - Files should contain normalized coordinate data
2. Run training script:
   ```
   without_cross_validation.py
   ```

### Recording Gestures & Prediction
1. Launch the GUI:
   ```
   python gui_predictor.py
   ```
2. Using the Interface:
   - Click "Start Tracking" to begin recording
   - Move green object to draw gesture
   - Click "Stop Tracking" to finish recording
   - Click "Plot Data" to view visualization and get prediction
3. View Results:
   - Movement path displayed in real-time
   - Coordinate graph shown after recording
   - Predicted spell displayed on interface
   - Data automatically saved as CSV

## Components

### gui_predictor.py
Key features:
- Webcam feed display with real-time tracking
- Green object detection and path tracking
- Movement path visualization
- Data normalization and interpolation
- Automatic prediction after recording
- Plot generation for recorded data
- CSV export functionality

### full_data_rf_training.py
Functionality:
- Loads training data from organized directories
- Processes and normalizes coordinate data
- Extracts relevant features for classification
- Trains Random Forest model
- Saves model and associated files

## Data Processing
The system performs:
1. Real-time coordinate tracking
2. Normalization to [0,1] range
3. Interpolation to 100 points
4. Feature extraction:
   - Relative movements (dx, dy)
   - Speed calculations
   - Angle measurements
   - Statistical features

## Dependencies
- numpy: Data processing and calculations
- pandas: Data manipulation and CSV handling
- scikit-learn: Machine learning implementation
- opencv-python: Webcam interface and image processing
- Pillow: Image handling for GUI
- matplotlib: Data visualization
- scipy: Interpolation functions
- tkinter: GUI framework

## Model Files
Stored in rf_all_data_model_files/:
- model.pkl: Trained Random Forest classifier
- label_encoder.pkl: Encoder for spell labels
- scaler.pkl: Feature scaler for normalization

## Future Enhancements
- Multiple object tracking support
- Additional gesture types
- Real-time prediction during drawing
- Advanced visualization options
- Batch processing capability
- Model performance analytics
- Custom gesture recording

## Troubleshooting
- Ensure proper lighting for green object detection
- Check webcam permissions
- Verify all model files are present
- Confirm Python version compatibility
- Check all dependencies are installed

## License
This project is licensed under the MIT License.
