import litellm
from dotenv import load_dotenv

load_dotenv()

MODEL = "azure/DeepSeek-V3.2"
PROMPT = "Say only 'Hello'"

response = litellm.completion(model=MODEL, messages=[{"role": "user", "content": PROMPT}], max_tokens=100)

print(response.choices[0].message.content)
print(f"\nTokens: {response.usage.prompt_tokens} in / {response.usage.completion_tokens} out")
