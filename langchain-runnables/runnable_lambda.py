from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough,RunnableSequence,RunnableParallel, RunnableLambda
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

def word_counter(text):
    return len(text.split())


llm1 = HuggingFaceEndpoint(
    repo_id = "deepseek-ai/DeepSeek-V3.2",
    task = "text-generation"
) # type: ignore

model1 = ChatHuggingFace(llm = llm1)

prompt1 = PromptTemplate(
    template = " Write a joke about {topic}",
    input_variables=['topic']
)

parser = StrOutputParser()

prompt2 = PromptTemplate(
    template = "Explain the following Joke - {text}",
    input_variables = ['text']
)


joke_gen_chain = RunnableSequence(prompt1,model1,parser)

parallel_chain = RunnableParallel({
    'joke': RunnablePassthrough(),
    'count' : RunnableLambda(word_counter)
})

final_chain = RunnableSequence(joke_gen_chain,parallel_chain)

result = final_chain.invoke({'topic':'Indian Education'})

print(result)


