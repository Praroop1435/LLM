from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableParallel,RunnableSequence
from langchain_core.prompts import PromptTemplate

load_dotenv()

llm1 = HuggingFaceEndpoint(
    repo_id = "deepseek-ai/DeepSeek-V3.2",
    task = "text-generation"
) # type: ignore


llm2 = HuggingFaceEndpoint(
    repo_id = "unsloth/GLM-4.7-Flash-REAP-23B-A3B-GGUF",
    task = "text-generation"
) # type: ignore

model1 = ChatHuggingFace(llm = llm1)
model2 = ChatHuggingFace(llm = llm2)

prompt1 = PromptTemplate(
    template=(
        "Write a single LinkedIn post about {topic}.\n\n"
        "Constraints:\n"
        "- 120–150 words\n"
        "- Professional, first-person tone\n"
        "- No headings, no bullet points\n"
        "- No multiple options or templates\n"
        "- Output only the post text\n"
    ),
    input_variables=["topic"],
)

prompt2 = PromptTemplate(
    template = "Generate a tweet about the {topic}",
    input_variables=['topic']
)


parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'tweet' : RunnableSequence(prompt2,model1,parser),
    'linkedin' : RunnableSequence(prompt1,model1,parser)
})

result = parallel_chain.invoke({"topic":"AI"})
print(result)