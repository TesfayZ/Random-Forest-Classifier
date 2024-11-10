import requests

# URL of the deployed model API
url = "https://random-forest-classifier-5a80c73c3027.herokuapp.com/predict"

# Input data 
data = {
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
}

# Send the POST request
response = requests.post(url, json=data)

# Check the status code and print the response
if response.status_code == 200:
    print("Prediction received:", response.json())
else:
    print(f"Error: {response.status_code}, {response.text}")
