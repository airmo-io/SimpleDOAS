import requests

try:
    response = requests.get("http://hitran.org")
    if response.status_code == 200:
        print("Connection to HITRAN is successful.")
    else:
        print("Connection to HITRAN failed with status code:", response.status_code)
except Exception as e:
    print("Error connecting to HITRAN:", e)