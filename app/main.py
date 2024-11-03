from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import numpy as np
import logging
app = FastAPI()


@app.get("/")
async def read_root():
    return {"message": "Welcome to the prediction API!"}


# Configure logging
logging.basicConfig(level=logging.INFO)

# Load your model and encoder
model = joblib.load("model/model.joblib")
encoder = joblib.load("model/encoder.joblib")
lb = joblib.load("model/label_binarizer.joblib")
print(f"Label Encoder Classes: {lb.classes_}")

# Define your categorical features and label
cat_features = [
    "workclass", "education", "marital-status", "occupation",
    "relationship", "race", "sex", "native-country"
]


# Define process_data function
def process_data(data, categorical_features, encoder):
    X = data.copy()

    # Encode categorical features
    X_encoded = encoder.transform(X[categorical_features])
    X = X.drop(columns=categorical_features)
    X = np.concatenate([X.values, X_encoded], axis=1)

    return X


# Pydantic model for the POST request body
class InferenceRequest(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    # Use alias for hyphenated field
    education_num: int = Field(..., alias="education-num")
    marital_status: str = Field(..., alias="marital-status")
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int = Field(..., alias="hours-per-week")
    native_country: str = Field(..., alias="native-country")

    class Config:
        # Allow population by field name, important for alias handling
        populate_by_name = True


@app.post("/predict")
async def predict(request: InferenceRequest):
    # Map the incoming request fields
    input_data = {
        "age": request.age,
        "workclass": request.workclass,
        "fnlgt": request.fnlgt,
        "education": request.education,
        "education-num": request.education_num,
        "marital-status": request.marital_status,
        "occupation": request.occupation,
        "relationship": request.relationship,
        "race": request.race,
        "sex": request.sex,
        "capital_gain": request.capital_gain,
        "capital_loss": request.capital_loss,
        "hours-per-week": request.hours_per_week,
        "native-country": request.native_country
    }

    try:
        # Convert the input to DataFrame for processing
        df_input = pd.DataFrame([input_data])
        # Process data
        X = process_data(df_input, cat_features, encoder=encoder)
        prediction = model.predict(X)

        # Convert predictions to original labels
        prediction_labels = lb.inverse_transform(prediction)

        return {"input": input_data, "prediction": prediction_labels.tolist()}

    except Exception as e:
        logging.error(f"Error during prediction: {str(e)}")
        raise HTTPException(status_code=500,
                            detail=f"Internal Server Error: {str(e)}")


if __name__ == "__main__":

    # Create a DataFrame with similar input as test case to verify model
    test_data = pd.DataFrame([{
        "age": 40,
        "workclass": "Private",
        "fnlgt": 154374,
        "education": "HS-grad",
        "education-num": 9,
        "marital-status": "Married-civ-spouse",
        "occupation": "Machine-op-inspct",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital_gain": 15024,
        "capital_loss": 0,
        "hours-per-week": 40,
        "native-country": "United-States"
    }])

    # Process data
    X = process_data(test_data, cat_features, encoder=encoder)
    predicted = model.predict(X)
    prediction_labels = lb.inverse_transform(predicted)
    print("Test input predicted:", prediction_labels)
