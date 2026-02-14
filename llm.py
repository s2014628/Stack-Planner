from openai import OpenAI
import os

from dotenv import load_dotenv

load_dotenv()
api_key=os.getenv("EVAL_LLM_API_KEY")   
base_url=os.getenv("EVAL_LLM_BASE_URL")
model_name=os.getenv("EVAL_LLM_MODEL_NAME")
client = OpenAI(api_key=api_key, base_url=base_url)

response = client.chat.completions.create(
  model=model_name,
  messages=[{"role": "user", "content": "Why is the sky blue?"}],
)

response = response.choices[0].message.content
print(response)