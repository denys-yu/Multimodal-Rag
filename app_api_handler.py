import os
import uvicorn
import boto3
import json

from fastapi import FastAPI
from mangum import Mangum
from pydantic import BaseModel
from query_model import QueryModel
from rag_app.query_rag import query_rag

# Retrieve the name of the worker Lambda function from environment variables
WORKER_LAMBDA_NAME = os.environ.get("WORKER_LAMBDA_NAME", None)

# Initialize the FastAPI application
app = FastAPI()

# Mangum provides an adapter to make FastAPI compatible with AWS Lambda
handler = Mangum(app)  # Entry point for AWS Lambda execution


# Pydantic model to validate incoming POST request payload for submitting a query
class SubmitQueryRequest(BaseModel):
    query_text: str  # The query text to process


@app.get("/")
def index():
    """
    Root endpoint that returns a simple greeting.

    Returns:
        dict: A simple hello world message.
    """
    return {"Hello": "World"}


@app.get("/get_query")
def get_query_endpoint(query_id: str) -> QueryModel:
    """
    Endpoint to retrieve a query result by its unique query ID.

    Args:
        query_id (str): The unique ID of the query.

    Returns:
        QueryModel: The query object retrieved from the database.
    """
    query = QueryModel.get_item(query_id)
    return query


@app.post("/submit_query")
def submit_query_endpoint(request: SubmitQueryRequest) -> QueryModel:
    """
    Endpoint to submit a new query for processing.

    If the WORKER_LAMBDA_NAME is set, the query will be processed asynchronously by invoking a worker Lambda.
    Otherwise, the query will be processed synchronously by the `query_rag` function.

    Args:
        request (SubmitQueryRequest): The query text submitted in the request body.

    Returns:
        QueryModel: The updated query object containing the response and sources.
    """
    # Create a new QueryModel instance with the submitted query text
    new_query = QueryModel(query_text=request.query_text)

    if WORKER_LAMBDA_NAME:
        # If a worker Lambda is configured, invoke it asynchronously
        new_query.put_item()
        invoke_worker(new_query)
    else:
        # If no worker Lambda is configured, process the query synchronously
        query_response = query_rag(request.query_text)  # Call the RAG system
        new_query.answer_text = query_response.response_text  # Update with the model's response
        new_query.sources = query_response.sources  # Add the sources used
        new_query.is_complete = True  # Mark the query as complete

        new_query.put_item()

    return new_query


def invoke_worker(query: QueryModel):
    """
    Invoke a worker Lambda function asynchronously to process a query.

    Args:
        query (QueryModel): The query object to be sent to the worker Lambda.
    """
    # Initialize the AWS Lambda client
    lambda_client = boto3.client("lambda")

    # Convert the query model to a dictionary payload
    payload = query.model_dump()

    # Invoke the worker Lambda function asynchronously
    response = lambda_client.invoke(
        FunctionName=WORKER_LAMBDA_NAME,  # Target worker Lambda function name
        InvocationType="Event",  # Asynchronous invocation
        Payload=json.dumps(payload),  # Serialize the payload to JSON
    )

    print(f"Worker Lambda invoked: {response}")


if __name__ == "__main__":
    """
    Run the FastAPI application locally for testing.
    The server will listen on port 8000.
    """
    port = 8000
    print(f"Running the FastAPI server on port {port}.")
    uvicorn.run("app_api_handler:app", host="0.0.0.0", port=port)
