import os
import time
import uuid
import boto3
from pydantic import BaseModel, Field
from typing import List, Optional
from botocore.exceptions import ClientError

# Retrieve the DynamoDB table name from environment variables
TABLE_NAME = os.environ.get("TABLE_NAME")


class QueryModel(BaseModel):
    """
    Represents a query model that handles query data and integrates with DynamoDB.

    Attributes:
        query_id (str): A unique identifier for the query, auto-generated using UUID.
        create_time (int): The timestamp when the query was created.
        query_text (str): The original query text submitted by the user.
        answer_text (Optional[str]): The response text generated for the query.
        sources (List[str]): A list of document sources used to generate the response.
        is_complete (bool): A flag indicating if the query processing is complete.
    """
    query_id: str = Field(default_factory=lambda: uuid.uuid4().hex)  # Generate unique ID
    create_time: int = Field(default_factory=lambda: int(time.time()))  # Current timestamp
    query_text: str  # Input query text
    answer_text: Optional[str] = None  # Optional response text
    sources: List[str] = Field(default_factory=list)  # List of sources used in the response
    is_complete: bool = False  # Status flag for query completion

    @classmethod
    def get_table(cls: "QueryModel") -> boto3.resource:
        """
        Retrieve the DynamoDB table resource.

        Returns:
            boto3.resource: The DynamoDB table object.
        """
        dynamodb = boto3.resource("dynamodb")
        return dynamodb.Table(TABLE_NAME)

    def put_item(self):
        """
        Insert or update the current query model instance into the DynamoDB table.
        Handles errors and prints responses.
        """
        # Convert the QueryModel instance to a DynamoDB-compatible item
        item = self.as_ddb_item()
        try:
            response = QueryModel.get_table().put_item(Item=item)  # Insert item into the table
            print(response)
        except ClientError as e:
            print("ClientError", e.response["Error"]["Message"])  # Log client error message
            raise e  # Reraise exception for further handling

    def as_ddb_item(self):
        """
        Convert the QueryModel instance to a dictionary compatible with DynamoDB.
        Excludes any fields with `None` values.

        Returns:
            dict: A dictionary representation of the model for DynamoDB.
        """
        # Exclude None values from the dictionary
        item = {k: v for k, v in self.dict().items() if v is not None}
        return item

    @classmethod
    def get_item(cls: "QueryModel", query_id: str) -> "QueryModel":
        """
        Retrieve a query item from the DynamoDB table by its unique query ID.

        Args:
            query_id (str): The unique identifier for the query.

        Returns:
            QueryModel: An instance of QueryModel populated with data from DynamoDB.
                        Returns None if the item is not found.
        """
        try:
            # Get the item from the table using the query ID as the key
            response = cls.get_table().get_item(Key={"query_id": query_id})
        except ClientError as e:
            print("ClientError", e.response["Error"]["Message"])  # Log client error message
            return None

        # Check if the item exists in the response
        if "Item" in response:
            item = response["Item"]
            return cls(**item)  # Convert the DynamoDB item to a QueryModel instance
        else:
            return None  # Return None if the item does not exist
