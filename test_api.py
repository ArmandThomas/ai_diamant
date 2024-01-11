import requests

# URL of the FastAPI endpoint
url = "http://127.0.0.1:8000/predict"

# Example game state data



# Send a POST request
response = requests.post(url, json=data)

# Print the response
print("Status Code:", response.status_code)
print("Response:", response.json())
