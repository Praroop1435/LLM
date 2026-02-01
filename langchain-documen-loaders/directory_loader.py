from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

loader = DirectoryLoader(
    path = "langchain-documen-loaders/books",
    glob = "*.pdf",
    loader_cls= PyPDFLoader # type: ignore
)

docs = loader.load()

print(docs[4].page_content)