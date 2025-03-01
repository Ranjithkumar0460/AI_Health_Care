from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os


# Extract data from the PDF
def load_pdf(data):
    loader = DirectoryLoader(data, glob="*.pdf", loader_cls=PyPDFLoader)

    documents = loader.load()

    return documents


# Create text chunks
def text_split(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    text_chunks = text_splitter.split_documents(extracted_data)

    return text_chunks


# download embedding model
def download_hugging_face_embeddings():
    os.environ["GOOGLE_API_KEY"] = "AIzaSyDtYr7GWlJrBuL6qK-rOsa3D-8T-V1RDyw"
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    return embeddings
