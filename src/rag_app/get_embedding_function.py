from langchain_aws import BedrockEmbeddings


def get_embedding_function():
    """
    Initialize and return the embedding function using AWS Bedrock.

    BedrockEmbeddings is a wrapper around AWS Bedrock's embedding models.
    This function provides embeddings for text data, which are essential for
    vector databases or NLP tasks like search and retrieval.

    Returns:
        BedrockEmbeddings: An instance of AWS Bedrock's embedding model.
    """
    # Initialize the BedrockEmbeddings model
    embeddings = BedrockEmbeddings()
    return embeddings
