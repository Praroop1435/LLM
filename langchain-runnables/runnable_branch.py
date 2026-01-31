from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableSequence, RunnableParallel, RunnableLambda, RunnableBranch
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm1 = HuggingFaceEndpoint(
    repo_id = "MiniMaxAI/MiniMax-M2.1",
    task = "text-generation"
) # type: ignore

model1 = ChatHuggingFace(llm = llm1)


parser = StrOutputParser()

prompt1 = PromptTemplate(
    template = "Write a detailed report on {topic}",
    input_variables = ['topic']
)

prompt2 = PromptTemplate(
    template = "Summarize the following text \n {text}",
    input_variables= ['text']
)

report_gen_chain = RunnableSequence(prompt1, model1, parser)

branch_chain = RunnableBranch(
    (lambda x: len(x.split()) > 1100 , RunnableSequence(prompt2,model1,parser)), # type: ignore
    (RunnablePassthrough())
) 

final_chain = RunnableSequence(report_gen_chain,branch_chain)

result = final_chain.invoke({'topic': "Russia Vs Ukraine"})
print(result)


