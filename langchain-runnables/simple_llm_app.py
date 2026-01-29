from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

# Initialize the LLM
llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V3.2",
    task="text-generation"
)  # type: ignore

model = ChatHuggingFace(llm = llm)

# Create a Prompt Template
prompt = PromptTemplate(
    input_variables=["topic"],
    template="Suggest a catchy blog title about {topic}."
)

# Format prompt
formatted_prompt = prompt.format(topic="Cricket")

# Call the LLM
blog_title = model.invoke(formatted_prompt)

print("Generated Blog Title:", blog_title)
