from fastapi.testclient import TestClient
from main import app


client = TestClient(app)


def test_get_root():
    """Test the GET method on the root path."""
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the prediction API!"}


def test_predict_above_50():
    """Test the POST method for input that predicts salary >50K."""
    response = client.post("/predict", json={
        "age": 45,
        "workclass": "Private",
        "fnlgt": 214567,
        "education": "Doctorate",
        "education-num": 16,
        "marital-status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "hours-per-week": 40,
        "capital_gain": 50000,
        "capital_loss": 0,
        "native-country": "United-States"
    })

    # Print the response JSON to debug if there's an error
    print(response.json())

    assert response.status_code == 200
    # Check if prediciton is the label for above $50k
    assert response.json().get("prediction") == [1]


def test_predict_below_50():
    """Test the POST method for input that predicts salary <=50K."""
    response = client.post("/predict", json={
        "age": 25,
        "workclass": "Private",
        "fnlgt": 223577,
        "education": "Bachelors",
        "education-num": 9,
        "marital_status": "Married-civ-spouse",
        "occupation": "Tech-support",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "hours_per_week": 30,
        "capital_gain": 0,
        "capital_loss": 0,
        "native_country": "United-States"
    })

    # Print the response JSON to debug if there's an error
    print(response.json())

    assert response.status_code == 200
    # Check if prediciton is the label for above $50k
    assert response.json().get("prediction") == [0]
