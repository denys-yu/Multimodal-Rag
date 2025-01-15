from dataclasses import dataclass
from typing import List
from langchain.prompts import ChatPromptTemplate
from langchain_aws import ChatBedrock
from get_chroma_db import get_chroma_db

# Define a prompt template for responding to user queries. The response will suggest contacting support
# if the provided context is insufficient for generating an answer.
PROMPT_TEMPLATE = """
If the context is not enough for an answer, add contact information to the response and advise to contact 
support 'support@airobotics.com'
Do not include wording like: "The context does not provide enough information" into the answer.
Answer the question based only on the following context:


{context}

---

Answer the question based on the above context: {question}
"""

# Define the Bedrock model ID for querying the Amazon Bedrock Claude model
BEDROCK_MODEL_ID = "anthropic.claude-3-haiku-20240307-v1:0"


# Define a data class to hold the query response details
@dataclass
class QueryResponse:
    query_text: str  # Original user query text
    response_text: str  # Model's response to the query
    sources: List[str]  # List of document IDs or sources used to generate the response


def query_rag(query_text: str) -> QueryResponse:
    """
    Function to query a Retrieval-Augmented Generation (RAG) system and return a structured response.

    Args:
        query_text (str): The user input or question to process.

    Returns:
        QueryResponse: A data class containing the query, response, and document sources.
    """
    # Initialize the Chroma database for document retrieval
    db = get_chroma_db()

    # Retrieve top-k similar documents based on the query text
    results = db.similarity_search_with_score(query_text, k=5)

    # Group context by content type
    text_context = []
    table_context = []
    image_context = []

    for doc, _score in results:
        content_type = doc.metadata.get("type", "text")
        if content_type == "text":
            text_context.append(doc.page_content)
        elif content_type == "table":
            table_context.append(doc.page_content)
        elif content_type == "image":
            # For images, include a placeholder or base64-encoded image
            image_context.append(f"Image (Base64): {doc.page_content}")

    # Combine the grouped content into a single context
    context_text = (
            "### Text:\n" + "\n\n".join(text_context) +
            "\n\n### Tables:\n" + "\n\n".join(table_context) +
            "\n\n### Images:\n" + "\n\n".join(image_context)
    )

    # Generate the prompt template with the grouped context and user question
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    # Debugging: Print the prompt for verification
    print("Generated Prompt:")
    print(prompt)

    # Initialize the Amazon Bedrock model (Claude) for processing the prompt
    try:
        model = ChatBedrock(model_id=BEDROCK_MODEL_ID)
        response = model.invoke(prompt)
        response_text = response.content  # Extract the response content from the model output
    except Exception as e:
        response_text = f"Error generating response: {str(e)}"

    # Extract the document sources (metadata "id") from the retrieved results
    sources = [doc.metadata.get("id", "unknown") for doc, _score in results]

    # Debugging: Print the response and sources
    print("Response:", response_text)
    print("Sources:", sources)

    # Return the response encapsulated in the QueryResponse data class
    return QueryResponse(
        query_text=query_text, response_text=response_text, sources=sources
    )


if __name__ == "__main__":
    # Example query for testing the RAG pipeline
    query_rag("What is AI Robotics?")
