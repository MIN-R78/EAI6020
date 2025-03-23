#Title: EAI6020 MIN LI MAR 16

import requests
import json

# test
test_data = {
    "user_id": "3",  # ID
    "item_id": "15"   # ID
}

response = requests.post(
    'http://localhost:5000/predict',
    headers={'Content-Type': 'application/json'},
    data=json.dumps(test_data)
)

print("Status Code:", response.status_code)
print("Response Body:", response.json())