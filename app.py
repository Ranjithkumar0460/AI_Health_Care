from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain.vectorstores import Pinecone
import pinecone
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from src.prompt import *
import os
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# %pip install langchain_google_genai
from langchain_google_genai import ChatGoogleGenerativeAI

# If we already have an index we can load it like this

from langchain.vectorstores import Pinecone as LangchainPinecone
import os


app = Flask(__name__)


load_dotenv()

os.environ["GOOGLE_API_KEY"] = "AIzaSyBWHApDIq_ABuhoNjVQEG1lSRBo0V8fSxA"
PINECONE_API_KEY = (
    "pcsk_2GNsv2_2eHrSjryqVXodVq8v6QuJosEDV3TfdoTuoNZB7wkx2Mqjq5t6EiPt9v8EYoLuBf"
)
PINECONE_API_ENV = "gcp-starter"

index_name = "medicalbot"
os.environ["PINECONE_API_KEY"] = (
    "pcsk_2GNsv2_2eHrSjryqVXodVq8v6QuJosEDV3TfdoTuoNZB7wkx2Mqjq5t6EiPt9v8EYoLuBf"
)
docsearch = LangchainPinecone.from_existing_index(
    index_name, download_hugging_face_embeddings()
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

prompt_template = (
    "Use the following pieces of information to answer the question."
    "If you don't know the answer, just say that you don't know, don't try to make up an answer."
    "Use three sentences maximum and keep the answer concise"
    "\n\n"
    "{context}"
)
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.9, max_tokens=500)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", prompt_template),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)


# @app.route("/")
# def index():
#     return render_template("chat.html")


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    response = rag_chain.invoke({"input": input})
    print(response["answer"])
    print("Response : ", response["answer"])
    return str(response["answer"])


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
