from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("langchain-documen-loaders/dl-curriculum.pdf")

docs = loader.load()

print(len(docs))