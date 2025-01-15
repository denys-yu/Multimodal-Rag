from query_model import QueryModel
from rag_app.query_rag import query_rag


def handler(event, context):
    """
    AWS Lambda handler function to process incoming events.

    Args:
        event (dict): The input event containing query data (passed from AWS Lambda).
        context: The AWS Lambda execution context (not used here).
    """
    # Parse the incoming event data into a QueryModel instance
    query_item = QueryModel(**event)
    # Invoke the RAG system with the query data
    invoke_rag(query_item)


def invoke_rag(query_item: QueryModel):
    """
    Invoke the Retrieval-Augmented Generation (RAG) system to process the query and update the query item.

    Args:
        query_item (QueryModel): The query object containing the input text.

    Returns:
        QueryModel: The updated query object with the RAG response.
    """
    # Call the query_rag function to process the query text and get the response
    rag_response = query_rag(query_item.query_text)

    # Update the QueryModel instance with the response details
    query_item.answer_text = rag_response.response_text  # Model's response text
    query_item.sources = rag_response.sources  # List of document sources used for the answer
    query_item.is_complete = True  # Mark the query processing as complete

    query_item.put_item()
    print(f"Item is updated: {query_item}")

    return query_item


def main():
    """
    Main function for local testing of the RAG system invocation.
    """
    print("Running example RAG call.")

    # Create a sample query object with a test query
    query_item = QueryModel(
        query_text="What kind of robots do you have?"
    )

    # Invoke the RAG system and print the response
    response = invoke_rag(query_item)
    print(f"Received: {response}")


if __name__ == "__main__":
    """
    Run the script locally for testing.
    """
    main()
