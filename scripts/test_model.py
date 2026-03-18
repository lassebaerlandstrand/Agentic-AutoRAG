import litellm
from dotenv import load_dotenv

load_dotenv()

MODEL = "bedrock/global.anthropic.claude-haiku-4-5-20251001-v1:0"
PROMPT = "Say hello in one short sentence."

response = litellm.completion(
    model=MODEL, messages=[{"role": "user", "content": PROMPT}], temperature=0.0, max_tokens=1
)

print(response.choices[0].message.content)
print(f"\nTokens: {response.usage.prompt_tokens} in / {response.usage.completion_tokens} out")
