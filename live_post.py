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

# Send POST request to the API
response = requests.post(url, json=data)

# Check if the status code indicates success
if response.status_code == 200:
    result = response.json()
    prediction = result['prediction'][0]
    print(f"Prediction received: {prediction}")
else:
    # Print the error status code and message
    print(f"Error: {response.status_code}, {response.text}")

# Display both result and status code
print("Status Code:", response.status_code)
print("Response Content:", response.json())
