import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()  # MUST be before os.environ access

hf_token = os.getenv("HF_TOKEN")
assert hf_token is not None, "HF_TOKEN not found in environment"

client = InferenceClient(api_key=hf_token)

completion = client.chat.completions.create(
    model="MiniMaxAI/MiniMax-M2.1",
    messages=[
        {"role": "user", "content": "What is the capital of France?"}
    ],
)

print(completion.choices[0].message.content)
