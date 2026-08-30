# LangChain Imports
from langchain_community.document_loaders import PyPDFLoader
import tiktoken

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# 1. READ PDF AND COUNT RAW TOKENS (LANGCHAIN)
def get_pdf_tokens(file_path: str, model_name: str = "gpt-4o") -> tuple[str, int, int]:
    """Loads a PDF and returns its full text content, exact token count, and word count."""
    # Load PDF text using LangChain community loader
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    
    # Merge pages into a single string
    full_text = "\n\n".join([doc.page_content for doc in docs])
    
    # Count tokens using the target model's encoding structure
    encoding = tiktoken.encoding_for_model(model_name)
    token_count = len(encoding.encode(full_text))
    word_count = len(full_text.split())
    
    return full_text, token_count, word_count


# 4. EXECUTION DRIVER WITH METADATA TRACKING
if __name__ == "__main__":
    # Path to your targeted PDF file
    pdf_path = r"data/pdf/Introduction to Algorithms - 3rd Edition.pdf" 
    
    print("--- Phase 1: LangChain Document Parsing ---")
    text_content, raw_tokens, raw_words = get_pdf_tokens(pdf_path, model_name="gpt-4.1")
    print(f"Successfully read PDF.")
    print(f"📄 Total Static PDF Tokens: {raw_tokens}")
    print(f"📝 Total Static PDF Words: {raw_words}\n")

