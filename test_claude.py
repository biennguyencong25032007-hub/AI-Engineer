import requests

API_KEY = "dán_api_key_của_bạn_vào_đây"

url = "https://api.taphoaapi.com/v1/chat/completions"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

data = {
    "model": "claude-3-sonnet",
    "messages": [
        {"role": "user", "content": "Viết code Python in Hello World"}
    ]
}

response = requests.post(url, headers=headers, json=data)

print(response.json())