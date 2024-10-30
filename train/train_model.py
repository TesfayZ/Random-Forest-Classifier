# Import necessary libraries
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# Functions

def load_data(filepath):
    """Load the dataset from a CSV file."""
    data = pd.read_csv(filepath)
    # Remove whitespace from columns
    data.columns = data.columns.str.strip()  
    return data

def process_data(data, categorical_features, label, training=True, encoder=None, lb=None):
    """Process the data by encoding categorical features and the label."""
    X = data.drop(columns=[label])
    y = data[label]

    if training:
        # Initialize the encoders for categorical features and label
        encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
        lb = LabelEncoder()

        # Fit and transform the categorical features
        X_encoded = encoder.fit_transform(X[categorical_features])
        X = X.drop(columns=categorical_features)
        X = np.concatenate([X.values, X_encoded], axis=1)

        # Fit and transform the label
        y = lb.fit_transform(y)

        print(f"Training: Encoded features shape: {X.shape}, target shape: {y.shape}")

    else:
        # Transform the data using the existing encoder and label binarizer
        X_encoded = encoder.transform(X[categorical_features])
        X = X.drop(columns=categorical_features)
        X = np.concatenate([X.values, X_encoded], axis=1)
        y = lb.transform(y)

        print(f"Inference: Encoded features shape: {X.shape}, target shape: {y.shape}")

    return X, y, encoder, lb


def train_model(X, y):
    """Train a random forest classifier on the data."""
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)
    return model

def save_model(model, encoder, lb, model_filepath, encoder_filepath, lb_filepath):
    """Save the trained model, encoder, and label binarizer."""
    joblib.dump(model, model_filepath)
    joblib.dump(encoder, encoder_filepath)
    joblib.dump(lb, lb_filepath)

def load_model(model_filepath, encoder_filepath, lb_filepath):
    """Load the trained model, encoder, and label binarizer."""
    model = joblib.load(model_filepath)
    encoder = joblib.load(encoder_filepath)
    lb = joblib.load(lb_filepath)
    return model, encoder, lb

def evaluate_model(model, X, y):
    """Evaluate the model and print classification metrics."""
    predictions = model.predict(X)
    accuracy = accuracy_score(y, predictions)
    precision = precision_score(y, predictions)
    recall = recall_score(y, predictions)
    f1 = f1_score(y, predictions)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    return accuracy, precision, recall, f1


def compute_slice_metrics(model, X_test, y_test, feature_name, feature_index):
    """
    Computes performance metrics for each unique value of the specified feature.
    
    Parameters:
        model: The trained model to evaluate.
        X_test: The test features as a NumPy array.
        y_test: The true labels for the test features.
        feature_name: The name of the categorical feature to slice by.
        feature_index: The index of the categorical feature.

    Returns:
        metrics: A DataFrame containing metrics for each unique value.
    """
    
     # Use indexing to get unique values
    unique_values = np.unique(X_test[:, feature_index]) 
    metrics = []

    for value in unique_values:
        # Slice the test data for the current value
        slice_indices = np.where(X_test[:, feature_index] == value)
        slice_data = X_test[slice_indices]
        slice_labels = y_test[slice_indices]

        # Make predictions
        predictions = model.predict(slice_data)

        # Calculate metrics
        accuracy = accuracy_score(slice_labels, predictions)
        precision = precision_score(slice_labels, predictions, average='weighted', zero_division=0)
        recall = recall_score(slice_labels, predictions, average='weighted', zero_division=0)
        f1 = f1_score(slice_labels, predictions, average='weighted', zero_division=0)

        metrics.append({
            'value': value,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
        })

    # Convert to DataFrame for easier output
    metrics_df = pd.DataFrame(metrics)
    return metrics_df


if __name__ == "__main__":
    # Filepath for the data
    data_filepath = os.path.join("..", "data", "census.csv")

    # Load the data
    data = load_data(data_filepath)
    #print("columns:", data.columns)
    
    # Define categorical features and label column
    cat_features = [
        "workclass", "education", "marital-status", "occupation",
        "relationship", "race", "sex", "native-country"
    ]
    label = "salary"
    
    # Split the data into train and test sets
    train, test = train_test_split(data, test_size=0.30, random_state=42)
    
    # Process the training data
    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label=label, training=True
    )
    
    # Process the test data using the trained encoder and label binarizer
    X_test, y_test, _, _ = process_data(
        test, categorical_features=cat_features, label=label, training=False,
        encoder=encoder, lb=lb
    )
    
    # Train the model
    model = train_model(X_train, y_train)

    # Compute and print slice metrics for 'education'
    feature_name = 'education'
    # Get index from the categorical features list because data is now ndarray
    feature_index = cat_features.index(feature_name)  

    # Call the modified compute_slice_metrics function
    metrics_df = compute_slice_metrics(model, X_test, y_test, feature_name, feature_index)

    # Output to a file
    output_file_path = "slice_output.txt"
    with open(output_file_path, "w") as f:
        f.write(metrics_df.to_string(index=False))
    
    # Save the model and encoders
    save_model(model, encoder, lb, "../model/model.joblib", "../model/encoder.joblib", "../model/label_binarizer.joblib")
    
    # Evaluate the model
    print("Evaluation on Test Set:")
    evaluate_model(model, X_test, y_test)
