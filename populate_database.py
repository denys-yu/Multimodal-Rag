import argparse
import os
import shutil
import fitz  # pymupdf for PDF processing
import base64
from langchain.schema.document import Document
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from src.rag_app.get_embedding_function import get_embedding_function
from pydantic import BaseModel

# Define paths for ChromaDB storage and source documents
CHROMA_PATH = "src/data/chroma"  # Path to store Chroma vector database
DATA_SOURCE_PATH = "src/data/source"  # Path to the directory containing source PDF files


def extract_pdf_content(file_path):
    """
    Extract text, images, and tables from a PDF file.

    Args:
        file_path (str): Path to the PDF file.

    Returns:
        list[Document]: A list of Document objects containing text, images, and table data.
    """
    documents = []
    pdf_document = fitz.open(file_path)

    for page_num in range(len(pdf_document)):
        page = pdf_document[page_num]

        # Extract text content
        text = page.get_text("text")
        if text.strip():
            documents.append(Document(page_content=text, metadata={"source": file_path, "page": page_num + 1}))

        # Extract images
        image_list = page.get_images(full=True)
        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = pdf_document.extract_image(xref)
            image_bytes = base_image["image"]
            # Encode image bytes to base64 to make it a string
            image_base64 = base64.b64encode(image_bytes).decode('utf-8')
            image_metadata = {
                "source": file_path,
                "page": page_num + 1,
                "type": "image",
                "image_index": img_index,
            }
            documents.append(Document(page_content=image_base64, metadata=image_metadata))

        # Extract tables
        blocks = page.get_text("dict")["blocks"]
        for block_index, block in enumerate(blocks):
            lines = block.get("lines", [])
            if len(lines) > 1 and all(len(line["spans"]) > 0 for line in lines):  # Simple heuristic for tables
                table_content = "\n".join(["\t".join(span["text"] for span in line["spans"]) for line in lines])
                table_metadata = {
                    "source": file_path,
                    "page": page_num + 1,
                    "type": "table",
                    "table_index": block_index,
                }
                documents.append(Document(page_content=table_content, metadata=table_metadata))

    pdf_document.close()
    return documents


def load_documents():
    """
    Load PDF documents and extract their multimodal content.

    Returns:
        list[Document]: A list of Document objects containing text, images, and tables.
    """
    documents = []

    for file_name in os.listdir(DATA_SOURCE_PATH):
        file_path = os.path.join(DATA_SOURCE_PATH, file_name)

        if file_name.lower().endswith(".pdf"):
            print(f"Processing {file_name}...")
            documents.extend(extract_pdf_content(file_path))

    return documents


def split_documents(documents: list[Document]):
    """
    Split loaded documents into smaller text chunks for processing.

    Args:
        documents (list[Document]): List of documents to split.

    Returns:
        list[Document]: A list of smaller text chunks from the documents.
    """
    max_chunk_size = 5000  # Split to ensure no chunk exceeds the max allowed size

    # Adjust splitter configuration to produce smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_chunk_size,  # Reduced chunk size
        chunk_overlap=200,  # Overlap between chunks to preserve context
        length_function=len,  # Function to calculate text length
        is_separator_regex=False,  # Treat the separator as a plain string
    )

    return text_splitter.split_documents(documents)


def add_to_chroma(documents: list[Document]):
    """
    Add or update document chunks into the Chroma vector database.

    Args:
        documents (list[Document]): List of document chunks to add.
    """
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=get_embedding_function())

    # Assign unique IDs to the document chunks
    for doc in documents:
        source = doc.metadata.get("source")
        page = doc.metadata.get("page")
        doc_type = doc.metadata.get("type", "text")
        doc.metadata["id"] = f"{source}:{page}:{doc_type}:{hash(doc.page_content)}"

    # Filter new documents that don't already exist in the database
    existing_items = db.get(include=[])  # Fetch all IDs from the database
    existing_ids = set(existing_items["ids"])
    new_documents = [doc for doc in documents if doc.metadata["id"] not in existing_ids]

    # Add new documents to the database
    if new_documents:
        db.add_documents(new_documents, ids=[doc.metadata["id"] for doc in new_documents])
        print(f"Added {len(new_documents)} new documents to ChromaDB.")
    else:
        print("No new documents to add.")


def clear_database():
    """
    Delete the Chroma vector database directory to reset the database.
    """
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)


def main():
    """
    Main function to load, process, and add documents to the Chroma vector database.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args = parser.parse_args()

    if args.reset:
        print("Clearing Database")
        clear_database()

    # Load documents, split into chunks, and add them to the database
    documents = load_documents()
    chunks = split_documents(documents)
    add_to_chroma(chunks)


if __name__ == "__main__":
    main()
