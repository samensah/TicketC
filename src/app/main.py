"""
Streamlit application for PDF processing using Ollama + LangChain with full-context approach.

This application allows users to upload a PDF, process the entire content,
and then ask questions about the content using phi4-mini.
"""

import streamlit as st
import logging
import os
import tempfile
import shutil
import pdfplumber
import ollama
import warnings
import tiktoken

# Suppress torch warning
warnings.filterwarnings('ignore', category=UserWarning, message='.*torch.classes.*')

from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from typing import List, Any, Optional

# Set protobuf environment variable to avoid error messages
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

# Streamlit page configuration
st.set_page_config(
    page_title="Phi4-mini PDF Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)

# Function to count tokens
def count_tokens(string: str, encoding_name: str = "cl100k_base") -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def extract_model_names(models_info: Any) -> List[str]:
    """
    Extract model names from the provided models information.
    
    Args:
        models_info: Response from ollama.list()
        
    Returns:
        List[str]: A list of model names.
    """
    logger.info("Extracting model names from models_info")
    try:
        # The new response format returns a list of Model objects
        if hasattr(models_info, "models"):
            # Extract model names from the Model objects
            model_names = [model.model for model in models_info.models]
        else:
            # Fallback for any other format
            model_names = []
            
        logger.info(f"Extracted model names: {model_names}")
        return model_names
    except Exception as e:
        logger.error(f"Error extracting model names: {e}")
        return []

def extract_text_from_pdf(file_upload) -> str:
    """
    Extract full text content from an uploaded PDF file.
    
    Args:
        file_upload (st.UploadedFile): Streamlit file upload object containing the PDF.
        
    Returns:
        str: The full text content of the PDF.
    """
    logger.info(f"Extracting text from file upload: {file_upload.name}")
    temp_dir = tempfile.mkdtemp()
    
    path = os.path.join(temp_dir, file_upload.name)
    with open(path, "wb") as f:
        f.write(file_upload.getvalue())
        logger.info(f"File saved to temporary path: {path}")
        loader = UnstructuredPDFLoader(path)
        data = loader.load()
    
    # Join all document content into a single string
    full_content = "\n\n".join([doc.page_content for doc in data])
    logger.info(f"Extracted {len(full_content)} characters from PDF")
    
    # Count tokens in the full content
    content_tokens = count_tokens(full_content)
    logger.info(f"Full content contains {content_tokens} tokens")
    
    shutil.rmtree(temp_dir)
    logger.info(f"Temporary directory {temp_dir} removed")
    return full_content

def process_question(question: str, full_content: str, selected_model: str) -> dict:
    """
    Process a user question using the full document content and selected language model.
    
    Args:
        question (str): The user's question.
        full_content (str): The full text content of the PDF.
        selected_model (str): The name of the selected language model.
        
    Returns:
        dict: Contains the generated response and token statistics.
    """
    logger.info(f"Processing question: {question} using model: {selected_model}")
    
    # Initialize LLM - using phi4-mini by default or selected model
    llm = ChatOllama(model=selected_model)
    
    # Full context prompt template
    template = """You are an assistant analyzing a PDF document.

Here is the complete content of the document:

{document_content}

Based on the above content ONLY, please answer the following question:
{question}

Provide a detailed response based solely on the information available in the document.
"""
    
    # Get token counts
    template_tokens = count_tokens(template)
    question_tokens = count_tokens(question)
    content_tokens = count_tokens(full_content)
    total_tokens = content_tokens + template_tokens + question_tokens - count_tokens("{document_content}") - count_tokens("{question}")
    
    # Model context window sizes - phi4-mini has 128K context window
    model_context_sizes = {
        "phi4-mini": 131072,  # 128K tokens
        "llama3": 8192,
        "mistral": 8192,
        "mixtral": 32768
    }
    
    # Get context size for the selected model (default to 8192 if unknown)
    model_max_tokens = model_context_sizes.get(selected_model.split(':')[0].lower(), 8192)
    tokens_left = model_max_tokens - total_tokens
    
    # Check if content fits in model's context window
    if total_tokens > model_max_tokens:
        return {
            "response": f"⚠️ Warning: The document is too large for {selected_model}'s context window ({total_tokens} tokens vs {model_max_tokens} max tokens). Please use a model with a larger context window like phi4-mini (128K tokens).",
            "token_info": {
                "content_tokens": content_tokens,
                "total_tokens": total_tokens,
                "max_tokens": model_max_tokens,
                "fits_context": False
            }
        }
    
    # Create prompt
    prompt = ChatPromptTemplate.from_template(template)
    
    # Create chain
    chain = (
        {"document_content": lambda _: full_content, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    # Process the question
    response = chain.invoke(question)
    logger.info("Question processed and response generated")
    
    return {
        "response": response,
        "token_info": {
            "content_tokens": content_tokens,
            "total_tokens": total_tokens,
            "max_tokens": model_max_tokens,
            "fits_context": True,
            "tokens_left": tokens_left
        }
    }

@st.cache_data
def extract_all_pages_as_images(file_upload) -> List[Any]:
    """
    Extract all pages from a PDF file as images.
    
    Args:
        file_upload (st.UploadedFile): Streamlit file upload object containing the PDF.
        
    Returns:
        List[Any]: A list of image objects representing each page of the PDF.
    """
    logger.info(f"Extracting all pages as images from file: {file_upload.name}")
    pdf_pages = []
    with pdfplumber.open(file_upload) as pdf:
        pdf_pages = [page.to_image().original for page in pdf.pages]
    logger.info("PDF pages extracted as images")
    return pdf_pages

def clear_session_state() -> None:
    """
    Clear relevant session state variables.
    """
    logger.info("Clearing session state")
    keys_to_clear = ["pdf_pages", "file_upload", "full_content"]
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]
    
    st.success("Session cleared successfully.")
    logger.info("Session state cleared")
    st.rerun()

def main() -> None:
    """
    Main function to run the Streamlit application.
    """
    st.subheader("TICKETC", divider="gray", anchor=False)
    st.write("Upload an ABC form and ask questions")
    
    # Get available models
    models_info = ollama.list()
    available_models = extract_model_names(models_info)
    
    # Ensure phi4-mini is available and preferred
    if "phi4-mini:latest" in available_models:
        default_index = available_models.index("phi4-mini:latest")
    else:
        default_index = 0
    
    # Create layout
    col1, col2 = st.columns([1.5, 2])
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "full_content" not in st.session_state:
        st.session_state["full_content"] = None
    if "use_sample" not in st.session_state:
        st.session_state["use_sample"] = False
    
    # Model selection
    if available_models:
        selected_model = col2.selectbox(
            "Select a model (phi4-mini recommended for best results) ↓", 
            available_models,
            index=default_index,
            key="model_select"
        )
    else:
        st.error("No Ollama models found. Make sure Ollama is running and models are installed.")
        return
    
    # Add checkbox for sample PDF
    use_sample = col1.toggle(
        "Use sample PDF", 
        key="sample_checkbox"
    )
    
    # Clear data if switching between sample and upload
    if use_sample != st.session_state.get("use_sample"):
        if st.session_state["full_content"] is not None:
            st.session_state["full_content"] = None
            st.session_state["pdf_pages"] = None
        st.session_state["use_sample"] = use_sample
    
    if use_sample:
        # Use the sample PDF
        sample_path = "data/pdfs/sample/abc_sample.pdf"
        if os.path.exists(sample_path):
            if st.session_state["full_content"] is None:
                with st.spinner("Processing sample PDF..."):
                    loader = UnstructuredPDFLoader(file_path=sample_path)
                    data = loader.load()
                    st.session_state["full_content"] = "\n\n".join([doc.page_content for doc in data])
                    
                    # Open and display the sample PDF
                    with pdfplumber.open(sample_path) as pdf:
                        st.session_state["pdf_pages"] = [page.to_image().original for page in pdf.pages]
        else:
            st.error("Sample PDF file not found.")
    else:
        # Regular file upload
        file_upload = col1.file_uploader(
            "Upload a PDF file ↓", 
            type="pdf", 
            accept_multiple_files=False,
            key="pdf_uploader"
        )
        
        if file_upload:
            if st.session_state.get("full_content") is None or st.session_state.get("file_upload") != file_upload.name:
                with st.spinner("Processing uploaded PDF..."):
                    st.session_state["full_content"] = extract_text_from_pdf(file_upload)
                    # Store the uploaded file name in session state
                    st.session_state["file_upload"] = file_upload.name
                    # Extract and store PDF pages
                    st.session_state["pdf_pages"] = extract_all_pages_as_images(file_upload)
    
    # Display PDF if pages are available
    if "pdf_pages" in st.session_state and st.session_state["pdf_pages"]:
        # PDF display controls
        zoom_level = col1.slider(
            "Zoom Level", 
            min_value=100, 
            max_value=1000, 
            value=700, 
            step=50,
            key="zoom_slider"
        )
        
        # Display PDF pages
        with col1:
            with st.container(height=410, border=True):
                for page_image in st.session_state["pdf_pages"]:
                    st.image(page_image, width=zoom_level)
    
    # Display token information if content is loaded
    if st.session_state["full_content"]:
        content_tokens = count_tokens(st.session_state["full_content"])
        col1.write(f"Document size: {content_tokens:,} tokens")
        
        # Show whether document fits in context window
        phi4_mini_context = 131072  # 128K tokens
        current_model_context = 8192  # Default for most models
        
        # Determine model context size
        model_context_sizes = {
            "phi4-mini": 131072,
            "llama3": 8192,
            "mistral": 8192,
            "mixtral": 32768
        }
        
        # Get base model name
        base_model = selected_model.split(':')[0].lower()
        current_model_context = model_context_sizes.get(base_model, 8192)
        
        if content_tokens > current_model_context:
            col1.warning(f"⚠️ This document ({content_tokens:,} tokens) is too large for {selected_model}'s context window ({current_model_context:,} tokens)")
            if "phi4-mini:latest" in available_models and base_model != "phi4-mini:latest":
                col1.info("Consider switching to phi4-mini:latest for better results")
        else:
            col1.success(f"✅ Document fits in {selected_model}'s context window")
    
    # Clear data button
    clear_data = col1.button(
        "🗑️ Clear Data", 
        type="secondary",
        key="clear_button"
    )
    
    if clear_data:
        clear_session_state()
    
    # Chat interface
    with col2:
        message_container = st.container(height=500, border=True)
        
        # Display chat history
        for message in st.session_state["messages"]:
            avatar = "🤖" if message["role"] == "assistant" else "😎"
            with message_container.chat_message(message["role"], avatar=avatar):
                st.markdown(message["content"])
        
        # Chat input and processing
        if prompt := st.chat_input("Ask a question about the PDF...", key="chat_input"):
            try:
                # Add user message to chat
                st.session_state["messages"].append({"role": "user", "content": prompt})
                with message_container.chat_message("user", avatar="😎"):
                    st.markdown(prompt)
                
                # Process and display assistant response
                with message_container.chat_message("assistant", avatar="🤖"):
                    with st.spinner(":green[processing...]"):
                        if st.session_state["full_content"] is not None:
                            result = process_question(
                                prompt, st.session_state["full_content"], selected_model
                            )
                            
                            # Display token information
                            if not result["token_info"]["fits_context"]:
                                st.warning(result["response"])
                            else:
                                st.markdown(result["response"])
                                
                                # Optionally show token stats
                                with st.expander("Token Statistics"):
                                    st.write(f"Document size: {result['token_info']['content_tokens']:,} tokens")
                                    st.write(f"Total prompt size: {result['token_info']['total_tokens']:,} tokens")
                                    st.write(f"Model context window: {result['token_info']['max_tokens']:,} tokens")
                                    st.write(f"Tokens available for response: ~{result['token_info']['tokens_left']:,} tokens")
                        else:
                            st.warning("Please upload a PDF file first.")
                
                # Add assistant response to chat history
                if st.session_state["full_content"] is not None:
                    st.session_state["messages"].append(
                        {"role": "assistant", "content": result["response"]}
                    )
            
            except Exception as e:
                st.error(f"Error: {str(e)}", icon="⛔️")
                logger.error(f"Error processing prompt: {e}")
        else:
            if st.session_state["full_content"] is None:
                st.info("Upload a PDF file or use the sample PDF to begin chat...")


if __name__ == "__main__":
    main()
