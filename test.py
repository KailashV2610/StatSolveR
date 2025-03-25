import requests
import json

response = requests.post(
  url="https://openrouter.ai/api/v1/chat/completions",
  headers={
    "Authorization": "Bearer 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",
    "Content-Type": "application/json",
  },
  data=json.dumps({
    "model": "meta-llama/llama-3.3-70b-instruct:free",
    "messages": [
      {
        "role": "user",
        "content": "What is the meaning of life?"
      }
    ],
    
  })
)
result = response.json()
# print(response["choices"][0]["message"]["content"])
print(result["choices"][0]["message"]["content"])