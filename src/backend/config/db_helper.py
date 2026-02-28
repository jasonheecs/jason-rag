from qdrant_client import QdrantClient


def check_payload_index_exists(client: QdrantClient, collection_name: str, field_name: str) -> bool:
    try:
        collection_info = client.get_collection(collection_name)
        existing_indices = collection_info.payload_schema

        if field_name in existing_indices:
            print(f"Index exists for field '{field_name}'.")
            return True
        else:
            print(f"Index does not exist for field '{field_name}'.")
            return False

    except Exception as e:
        print(f"An error occurred: {e}")
        return False