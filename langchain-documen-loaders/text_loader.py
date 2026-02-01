from langchain_community.document_loaders import TextLoader
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()


llm1 = HuggingFaceEndpoint(
    repo_id = "MiniMaxAI/MiniMax-M2.1",
    task = "text-generation"
) # type: ignore

model1 = ChatHuggingFace(llm = llm1)

prompt = PromptTemplate(
    template='Write a summary for the following poem - \n {poem}',
    input_variables=['poem']
)

parser = StrOutputParser()

loader = TextLoader('langchain-documen-loaders/cricket.txt', encoding='utf-8')

docs = loader.load()

# print(type(docs))

# print(len(docs))

# print(docs[0].page_content)

# print(docs[0].metadata)

chain = prompt | model1 | parser

print(chain.invoke({'poem':docs[0].page_content}))
