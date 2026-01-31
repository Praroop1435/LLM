from dotenv import load_dotenv
import os

from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate

# Load env
load_dotenv()
assert os.getenv("HF_TOKEN"), "HF_TOKEN not found"

# Initialize MiniMax via HF Inference
llm = HuggingFaceEndpoint(
    repo_id="MiniMaxAI/MiniMax-M2.1",
    task="text-generation",
)  # type: ignore

model = ChatHuggingFace(llm=llm)

# Prompt template
prompt = PromptTemplate(
    input_variables=["topic"],
    template="Suggest a catchy blog title about {topic}."
)

# Format prompt
formatted_prompt = prompt.format(topic="Cricket")

# Invoke model
response = model.invoke(formatted_prompt)

print("Generated Blog Title:", response.content)
