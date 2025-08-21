# Magic Spells Classification

This project presents a **Random Forest–based machine learning pipeline** for classifying magic spell movements from coordinate data. It supports **training, evaluation, and prediction**, along with a **GUI application** for user-friendly single-file predictions.  

---

## 🚀 Features  

- **Data Loading**: Reads CSV files where each folder represents a class label.  
- **Preprocessing**: Normalizes coordinates and derives features (relative movement, speed, angles).  
- **Feature Extraction**: Computes statistical metrics (mean, standard deviation, etc.) for modeling.  
- **Model Training**: Trains a Random Forest classifier on processed features.  
- **Model Saving**: Stores the trained model, scaler, and label encoder for reuse.  
- **Prediction**:  
  - CLI: Predicts the class of a single CSV file.  
  - GUI: Provides an intuitive interface for predictions.  

---

## ⚙️ Setup  

### Prerequisites  
- Python **3.8+**  

### Install Dependencies  
```bash
pip install -r requirements.txt
