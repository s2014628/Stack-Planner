from openai import OpenAI

api_key = "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"
base_url = "https://openrouter.ai/api/v1"
model = "deepseek/deepseek-v3.2"

client = OpenAI(api_key=api_key, base_url=base_url)

print("Testing OpenRouter API...")
print(f"  Base URL: {base_url}")
print(f"  Model:    {model}")
print()

try:
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Say hello in one sentence."}],
        max_tokens=50,
        timeout=30,
    )
    content = response.choices[0].message.content
    print(f"SUCCESS! Response:\n  {content}")
except Exception as e:
    print(f"FAILED! {type(e).__name__}: {e}")
