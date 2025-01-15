from langchain_community.vectorstores import Chroma
import get_embedding_function

import shutil
import sys
import os
from langchain_community.vectorstores import Chroma
from get_embedding_function import get_embedding_function

# Set the default path for ChromaDB, can be overridden by environment variable 'CHROMA_PATH'
CHROMA_PATH = os.environ.get("CHROMA_PATH", "data/chroma")

# Check if the code is running in an image-based runtime (like AWS Lambda), defined by 'IS_USING_IMAGE_RUNTIME'
IS_USING_IMAGE_RUNTIME = bool(os.environ.get("IS_USING_IMAGE_RUNTIME", False))

# Singleton instance to ensure only one ChromaDB instance is created during execution
CHROMA_DB_INSTANCE = None


def get_chroma_db():
    """
    Initialize and return the singleton ChromaDB instance.
    Handles special cases for AWS Lambda environment by adjusting SQLite behavior.

    Returns:
        Chroma: An instance of the Chroma vector database.
    """
    global CHROMA_DB_INSTANCE
    if not CHROMA_DB_INSTANCE:

        # AWS Lambda runtime hack: Adjusts the SQLite module for compatibility
        # and ensures ChromaDB can write to the /tmp directory.
        if IS_USING_IMAGE_RUNTIME:
            __import__("pysqlite3")  # Import pysqlite3 as a replacement for sqlite3
            sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")  # Swap sqlite3 module
            copy_chroma_to_tmp()  # Copy the ChromaDB data to a writable tmp directory

        # Initialize the Chroma database instance with persistence and embedding function
        CHROMA_DB_INSTANCE = Chroma(
            persist_directory=get_runtime_chroma_path(),  # Path to store ChromaDB data
            embedding_function=get_embedding_function(),  # Function to generate embeddings
        )
        print(f"Init ChromaDB {CHROMA_DB_INSTANCE} from {get_runtime_chroma_path()}")

    return CHROMA_DB_INSTANCE


def copy_chroma_to_tmp():
    """
    Copy the ChromaDB data directory to /tmp in image-based runtime environments (e.g., AWS Lambda).
    Ensures write permissions for ChromaDB operations.
    """
    dst_chroma_path = get_runtime_chroma_path()

    # Create the destination directory if it does not exist
    if not os.path.exists(dst_chroma_path):
        os.makedirs(dst_chroma_path)

    # Check if the destination directory is empty
    tmp_contents = os.listdir(dst_chroma_path)
    if len(tmp_contents) == 0:
        print(f"Copying ChromaDB from {CHROMA_PATH} to {dst_chroma_path}")
        os.makedirs(dst_chroma_path, exist_ok=True)
        shutil.copytree(CHROMA_PATH, dst_chroma_path, dirs_exist_ok=True)  # Copy data
    else:
        print(f"ChromaDB already exists in {dst_chroma_path}")


def get_runtime_chroma_path():
    """
    Get the path where ChromaDB data will be stored at runtime.
    If running in an image-based environment, the path points to the /tmp directory for write access.

    Returns:
        str: Path to the ChromaDB persistence directory.
    """
    if IS_USING_IMAGE_RUNTIME:
        return f"/tmp/{CHROMA_PATH}"
    else:
        return CHROMA_PATH
