import pickle
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def load_data(base_dir):
    data = []
    labels = []
    
    # Iterate through each spell folder
    for spell_folder in sorted(os.listdir(base_dir)):
        spell_path = os.path.join(base_dir, spell_folder)
        
        if not os.path.isdir(spell_path):
            continue  # Skip non-folder files
        
        # Iterate through CSV files in the spell folder
        for csv_file in os.listdir(spell_path):
            file_path = os.path.join(spell_path, csv_file)
            
            if file_path.endswith('.csv'):
                # Load the CSV file
                df = pd.read_csv(file_path)
                data.append(df)
                labels.append(spell_folder)  # Use folder name as label
    
    return data, labels

def preprocess_coordinates(df):
    df['x'] = df['x'] - df['x'].mean()
    df['y'] = df['y'] - df['y'].mean()
    df['dx'] = df['x'].diff().fillna(0)
    df['dy'] = df['y'].diff().fillna(0)
    df['speed'] = np.sqrt(df['dx']**2 + df['dy']**2)
    df['angle'] = np.arctan2(df['dy'], df['dx']).fillna(0)
    
    # Additional Features
    df['acceleration'] = np.sqrt(df['dx'].diff().fillna(0)**2 + df['dy'].diff().fillna(0)**2)
    df['velocity'] = np.sqrt(df['dx']**2 + df['dy']**2)
    
    # Total distance traveled
    df['total_distance'] = df['speed'].cumsum()
    
    # Cumulative displacement
    df['cumulative_displacement'] = np.sqrt(df['x']**2 + df['y']**2)
    
    # Direction change (angle difference)
    df['angle_change'] = df['angle'].diff().fillna(0)
    
    # Time-related features
    df['time'] = np.arange(len(df))  # Assuming uniform time steps, otherwise replace with actual time
    
    # Geometric features
    bounding_box = [df['x'].min(), df['x'].max(), df['y'].min(), df['y'].max()]
    df['bounding_box_area'] = (bounding_box[1] - bounding_box[0]) * (bounding_box[3] - bounding_box[2])
    df['aspect_ratio'] = (bounding_box[1] - bounding_box[0]) / (bounding_box[3] - bounding_box[2])
    
    return df[['dx', 'dy', 'speed', 'angle', 'acceleration', 'velocity', 'total_distance', 'cumulative_displacement', 'angle_change', 'bounding_box_area', 'aspect_ratio']]

def extract_features(data_list):
    feature_set = []
    for df in data_list:
        features = {
            'dx_mean': df['dx'].mean(),
            'dx_std': df['dx'].std(),
            'dy_mean': df['dy'].mean(),
            'dy_std': df['dy'].std(),
            'speed_mean': df['speed'].mean(),
            'speed_std': df['speed'].std(),
            'angle_mean': df['angle'].mean(),
            'angle_std': df['angle'].std(),
            'acceleration_mean': df['acceleration'].mean(),
            'acceleration_std': df['acceleration'].std(),
            'velocity_mean': df['velocity'].mean(),
            'velocity_std': df['velocity'].std(),
            'total_distance': df['total_distance'].max(),
            'cumulative_displacement_mean': df['cumulative_displacement'].mean(),
            'angle_change_mean': df['angle_change'].mean(),
            'bounding_box_area': df['bounding_box_area'].max(),
            'aspect_ratio': df['aspect_ratio'].mean()
        }
        # Handle NaN or infinite values
        for key, value in features.items():
            if not np.isfinite(value):
                features[key] = 0  # Replace invalid values with 0 or another default
        feature_set.append(features)
    return pd.DataFrame(feature_set)


def train_random_forest(base_dir):
    # Load data and labels
    data_list, labels = load_data(base_dir)
    
    # Preprocess data
    processed_data = [preprocess_coordinates(df) for df in data_list]
    
    # Extract features
    X = extract_features(processed_data)
    
    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(labels)
    
    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Split data into train and test sets
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    
    # Train Random Forest model
    model = RandomForestClassifier(n_estimators=200, random_state=12)
    model.fit(X, y)
    
    # Evaluate model
    # y_pred = model.predict(X_test)
    # print("Accuracy:", accuracy_score(y_test, y_pred))
    # print(classification_report(y_test, y_pred, target_names=le.classes_))
    
    # # Confusion Matrix
    # cm = confusion_matrix(y_test, y_pred)
    # plt.figure(figsize=(10, 7))
    # sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
    # plt.xlabel('Predicted Labels')
    # plt.ylabel('True Labels')
    # plt.title('Confusion Matrix')
    # plt.show()
    
    return model, le, scaler


base_dir = 'All_Data - Copy'
model, label_encoder, scaler = train_random_forest(base_dir)


def save_model(model, label_encoder, scaler, save_dir="rf_all_data_model_files"):
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "model.pkl"), "wb") as model_file:
        pickle.dump(model, model_file)
    with open(os.path.join(save_dir, "label_encoder.pkl"), "wb") as le_file:
        pickle.dump(label_encoder, le_file)
    with open(os.path.join(save_dir, "scaler.pkl"), "wb") as scaler_file:
        pickle.dump(scaler, scaler_file)
def load_testing_data(test_dir):
    test_data = []
    test_files = []

    for csv_file in os.listdir(test_dir):
        file_path = os.path.join(test_dir, csv_file)
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
            test_data.append(df)
            test_files.append(csv_file)
    
    return test_data, test_files

def predict_testing_data(model, scaler, label_encoder, test_dir, output_csv="predictions.csv"):
    # Load and preprocess test data
    test_data_list, test_file_names = load_testing_data(test_dir)
    processed_test_data = [preprocess_coordinates(df) for df in test_data_list]
    X_test = extract_features(processed_test_data)
    X_test = scaler.transform(X_test)  # Standardize features

    # Predict
    predictions = model.predict(X_test)
    num_classes = len(label_encoder.classes_)
    
    # Convert predictions to one-hot encoding and save as CSV
    results = []
    for file_name, pred in zip(test_file_names, predictions):
        one_hot = np.zeros(num_classes, dtype=int)
        one_hot[pred] = 1
        results.append({"file_name": file_name, "prediction_tensor": one_hot.tolist()})
    
    # Save results to CSV
    pd.DataFrame(results).to_csv(output_csv, index=False)
    print(f"Predictions saved to {output_csv}")


# Save the trained model
save_model(model, label_encoder, scaler)
print("Model saved")