import requests

API_KEY = "sk-proj-1008b56f5cd34fc89fb7a58741dcf785"

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