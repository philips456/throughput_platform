import os
from dotenv import load_dotenv
import openai

# 🔐 Charger la clé depuis .env
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# 🔧 Initialiser OpenAI Client
client = openai.OpenAI(api_key=api_key)

# 📩 Envoyer un message
print("Testing OpenAI API...")
response = client.chat.completions.create(
    model="gpt-3.5-turbo", messages=[{"role": "user", "content": "Hello, ChatGPT!"}]
)

# 🖨️ Afficher la réponse
print("Response:\n", response.choices[0].message.content)
