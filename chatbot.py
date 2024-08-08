import os
from langchain_community.document_loaders import CSVLoader
from langchain_mistralai.embeddings import MistralAIEmbeddings
from langchain_pinecone import PineconeVectorStore, PineconeEmbeddings, Pinecone
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ai21 import AI21Embeddings
from time import sleep

os.environ["MISTRAL_API_KEY"] = "enIXLQGGLOW5YwDflhSpXY8Ms2N1ONIG"
os.environ["HF_TOKEN"] = "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"
os.environ["AI21_API_KEY"] = "qCltPgOsagPtZhlP2xeUIlkbYsmRzagH"
os.environ['PINECONE_API_KEY'] = 'a18aae9f-0bbe-4576-9c3c-c7e6abcb2f11'

index_name = "langchain-test-index"

# csv_files = os.listdir("./csv")

# docs = []

# loaders = []

# for csv_file in csv_files:
#     file_path = os.path.join("./csv", csv_file)
#     loader = CSVLoader(file_path)
#     loaders.append(loader)


# for loader in loaders:
#     docs.extend(loader.load())


loader = PyPDFLoader("THE_GLOBAL_AIRLINE_INDUSTRY.PDF")

docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1500,
    chunk_overlap = 150
)

splits = text_splitter.split_documents(docs)

embeddings = AI21Embeddings()

vectordb = DocArrayInMemorySearch.from_documents(splits, embeddings)

query = "Name an editor of the book 'The Global Airline Industry"

docs = vectordb.similarity_search(query)

print(docs[0].page_content)