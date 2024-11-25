import streamlit as st
import os
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
import tempfile
from pathlib import Path

class RAGChatbot:
    def __init__(self, pdf_content, groq_api_key: str):
        """Initialize the RAG chatbot with PDF content and Groq API key"""
        if not groq_api_key:
            raise ValueError("Groq API key is required")

        # Set up API key
        os.environ["GROQ_API_KEY"] = groq_api_key
        
        # Save PDF content to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(pdf_content)
            self.pdf_path = tmp_file.name

        try:
            # Initialize components
            with st.spinner('Loading and processing PDF...'):
                self.documents = self._load_and_split_pdf()
            with st.spinner('Creating vector store...'):
                self.vector_store = self._create_vector_store()
            with st.spinner('Setting up conversation chain...'):
                self.conversation_chain = self._setup_conversation_chain()
            self.chat_history = []
        except Exception as e:
            raise Exception(f"Failed to initialize chatbot: {str(e)}")
        finally:
            # Clean up temporary file
            if hasattr(self, 'pdf_path'):
                os.unlink(self.pdf_path)

    def _load_and_split_pdf(self) -> List:
        """Load and split PDF into chunks"""
        try:
            loader = PyPDFLoader(self.pdf_path)
            documents = loader.load()
            
            if not documents:
                raise ValueError("No text could be extracted from the PDF")
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
                separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
            )
            splits = text_splitter.split_documents(documents)
            
            if not splits:
                raise ValueError("No valid text chunks were created")
                
            return splits
        except Exception as e:
            raise Exception(f"Error processing PDF: {str(e)}")

    def _create_vector_store(self):
        """Create vector store using FAISS"""
        try:
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'}
            )
            
            if not self.documents:
                raise ValueError("No documents to create vector store")
                
            return FAISS.from_documents(self.documents, embeddings)
        except Exception as e:
            raise Exception(f"Error creating vector store: {str(e)}")

    def _setup_conversation_chain(self):
        """Set up the conversation chain"""
        try:
            llm = ChatGroq(
                temperature=0.7,
                model_name="llama-3.1-8b-instant",
                max_tokens=512,
                top_p=0.9,
                streaming=True
            )
            
            memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                output_key="answer"
            )
            
            return ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=self.vector_store.as_retriever(
                    search_kwargs={"k": 3}
                ),
                memory=memory,
                return_source_documents=True,
                verbose=True
            )
        except Exception as e:
            raise Exception(f"Error setting up conversation chain: {str(e)}")

    def ask_question(self, question: str) -> str:
        """Process a question and return the response"""
        if not question.strip():
            return "Please ask a valid question."
            
        try:
            response = self.conversation_chain({"question": question})
            
            answer = response["answer"]
            source_docs = response.get("source_documents", [])
            
            self.chat_history.append((question, answer))
            
            if source_docs:
                sources = "\n\nSources:\n" + "\n".join([
                    f"- Page {doc.metadata.get('page', 'N/A')}" 
                    for doc in source_docs
                ])
                return f"{answer}{sources}"
            
            return answer
        except Exception as e:
            return f"Error processing question: {str(e)}"

def initialize_session_state():
    """Initialize session state variables"""
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = None
    if 'messages' not in st.session_state:
        st.session_state.messages = []

def add_message(role, content):
    """Add a message to the chat history"""
    st.session_state.messages.append({"role": role, "content": content})

def main():
    st.set_page_config(page_title="RAG Chatbot", page_icon="🤖", layout="wide")
    
    # Initialize session state
    initialize_session_state()

    # Sidebar for API key and file upload
    with st.sidebar:
        st.title("Configuration")
        groq_api_key = st.text_input("Enter Groq API Key:", type="password")
        uploaded_file = st.file_uploader("Upload PDF", type="pdf")
        
        if uploaded_file and groq_api_key:
            if st.button("Process PDF"):
                with st.spinner("Processing PDF..."):
                    try:
                        pdf_content = uploaded_file.read()
                        st.session_state.chatbot = RAGChatbot(pdf_content, groq_api_key)
                        st.success("PDF processed successfully!")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")

    # Main chat interface
    st.title("📚 RAG Chatbot")
    
    # Display chat messages
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

    # Chat input
    if st.session_state.chatbot is not None:
        user_input = st.chat_input("Ask a question about your PDF...")
        if user_input:
            # Add user message
            add_message("user", user_input)
            
            # Get chatbot response
            with st.chat_message("assistant"):
                response = st.session_state.chatbot.ask_question(user_input)
                st.write(response)
                add_message("assistant", response)
    else:
        st.info("Please upload a PDF and enter your Groq API key to start chatting.")

if __name__ == "__main__":
    main()