import streamlit as st
import requests
import os
import time
from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
import google.api_core.exceptions

# Load environment variables
load_dotenv()

# Set Google API key
os.environ["GOOGLE_API_KEY"] = "AIzaSyBWHApDIq_ABuhoNjVQEG1lSRBo0V8fSxA"

# Create a selectbox or radio buttons for navigation
page = st.sidebar.radio("Choose a page", ["Home", "Chat", "Articles"])

# Conditional rendering based on the selected page
if page == "Home":
    st.title("Home Page")
    st.write("Welcome to the home page!")

elif page == "Chat":
    st.title("Chat Page")
    st.write("Welcome to the chat page!")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("Type your message..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Send user message to localhost:8080/get
        try:
            response = requests.post(
                "http://localhost:8080/get",  # Endpoint URL
                data={"msg": prompt},  # Form data
            )
            if response.status_code == 200:
                bot_response = response.text  # Get the response as plain text
            else:
                bot_response = f"Error: {response.status_code} - {response.text}"
        except requests.exceptions.RequestException as e:
            bot_response = f"Failed to connect to the server: {e}"

        # Display bot response in chat message container
        with st.chat_message("assistant"):
            st.markdown(bot_response)
        # Add bot response to chat history
        st.session_state.messages.append({"role": "assistant", "content": bot_response})

elif page == "Articles":
    st.title("RockyBot: News Research Tool 📈")
    st.sidebar.title("News Article URLs")

    urls = []
    for i in range(3):
        url = st.sidebar.text_input(f"URL {i+1}")
        urls.append(url)

    process_url_clicked = st.sidebar.button("Process URLs")
    file_path = "faiss_store"  # Directory to save FAISS index
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    main_placeholder = st.empty()
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro", temperature=0.9, max_tokens=500
    )

    if process_url_clicked:
        # Load data
        loader = UnstructuredURLLoader(urls=urls)
        main_placeholder.text("Data Loading...Started...✅✅✅")
        data = loader.load()
        # Split data
        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ".", ","], chunk_size=1000
        )
        main_placeholder.text("Text Splitter...Started...✅✅✅")
        docs = text_splitter.split_documents(data)
        # Create embeddings and save it to FAISS index
        vectorstore = FAISS.from_documents(docs, embeddings)
        main_placeholder.text("Embedding Vector Started Building...✅✅✅")
        time.sleep(2)

        # Save the FAISS index to disk
        vectorstore.save_local(file_path)
        main_placeholder.text("FAISS index saved successfully! ✅")

    query = main_placeholder.text_input("Question: ")
    if query:
        if os.path.exists(file_path):
            # Load the FAISS index from disk with dangerous deserialization allowed
            vectorstore = FAISS.load_local(
                file_path,
                embeddings,
                allow_dangerous_deserialization=True,  # Allow deserialization
            )
            chain = RetrievalQAWithSourcesChain.from_llm(
                llm=llm, retriever=vectorstore.as_retriever()
            )

            # Retry logic with exponential backoff
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    result = chain({"question": query}, return_only_outputs=True)
                    # Result will be a dictionary of this format --> {"answer": "", "sources": [] }
                    st.header("Answer")
                    st.write(result["answer"])

                    # Display sources, if available
                    sources = result.get("sources", "")
                    if sources:
                        st.subheader("Sources:")
                        sources_list = sources.split(
                            "\n"
                        )  # Split the sources by newline
                        for source in sources_list:
                            st.write(source)
                    break  # Exit the retry loop if successful
                except google.api_core.exceptions.ResourceExhausted as e:
                    if attempt < max_retries - 1:
                        wait_time = 2**attempt  # Exponential backoff
                        st.warning(
                            f"Resource exhausted. Retrying in {wait_time} seconds..."
                        )
                        time.sleep(wait_time)
                    else:
                        st.error("Resource exhausted. Please try again later.")
                except Exception as e:
                    st.error(f"An error occurred: {e}")
                    break
