from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableBranch , RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel,Field
from typing import Literal

load_dotenv()

llm1 = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
) # type: ignore

llm2 = HuggingFaceEndpoint(
    repo_id='deepseek-ai/DeepSeek-V3.2',
    task='text-generation'
) # type: ignore


model= ChatHuggingFace(llm = llm1)

class feedback(BaseModel):

    sentiment : Literal['positive','negative'] = Field(description= 'Give the sentiment of the feedback')

parser = StrOutputParser()
py_parser = PydanticOutputParser(pydantic_object=feedback)

prompt1 = PromptTemplate(
    template = "Classify the sentiment of the following text into negative or positive sentiment \n {feedback} \n{format_instruction}",
    input_variables = ['feedback'],
    partial_variables = {'format_instruction':py_parser.get_format_instructions()}
)

prompt2 = PromptTemplate(
    template = "Write an appropriate to this positive feedback \n {feedback}",
    input_variables = ['feedback']
)

prompt3 = PromptTemplate(
    template = "Write an appropriate to this negative feedback \n {feedback}",
    input_variables = ['feedback']
)



classifier_chain = prompt1 | model | py_parser


branch_chain = RunnableBranch(
    (lambda x:x.sentiment == "positive", prompt2 | model | parser),
    (lambda x:x.sentiment == "negative", prompt3 | model | parser),
    RunnableLambda(lambda x: "could not find the sentiment")
)



chain = classifier_chain | branch_chain

result = chain.invoke({'feedback':'The laptop is terrible'})

print(result)
chain.get_graph().print_ascii()

