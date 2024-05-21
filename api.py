import requests
import os
from dotenv import load_dotenv

load_dotenv()

from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage

api_key = os.environ["MISTRAL"]
model = "mistral-small-latest"

client = MistralClient(api_key=api_key)

messages = [
    ChatMessage(role="user", content="What is the best way to make dosa")
]

# With streaming
response = client.chat_stream(model=model, messages=messages)

for word in response:
    print(word.choices[0].delta.content)
