# pdf_parser.py

import logging
import PyPDF2

logger = logging.getLogger(__name__)

def extract_text_from_pdf(pdf_path):
    """Extract text from PDF file."""
    logger.info(f"Extracting text from PDF: {pdf_path}")
    text = ""
    
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            logger.info(f"PDF has {len(reader.pages)} pages")
            
            for i, page in enumerate(reader.pages):
                logger.debug(f"Processing page {i+1}")
                text += page.extract_text() + "\n"
                
        logger.info("Successfully extracted text from PDF")
        return text
    except Exception as e:
        logger.error(f"Error extracting text: {e}")
        return None