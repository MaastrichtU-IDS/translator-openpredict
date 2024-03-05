from abc import ABC, abstractmethod
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchText,
    PointStruct,
    UpdateResult,
    VectorParams,
    SearchParams,
)

from predict_drug_target.utils import log, COLLECTIONS


# Define an abstract class VectorDB
class VectorDB(ABC):
    def __init__(self, collections: list[dict[str, str | int]]):
        self.collections = collections
        pass

    @abstractmethod
    def add(self, collection_name: str, entity_id: str, vector: list[float], sequence: str | None = None) -> None:
        pass

    @abstractmethod
    def get(
        self, collection_name: str, search_input: str | None = None, search_field: str = "id", limit: int = 5
    ) -> list[Any]:
        pass

    @abstractmethod
    def search(self, collection_name: str, vector: str) -> list[tuple[str, float]]:
        pass


# https://qdrant.tech/documentation/quick-start
# More config: https://qdrant.tech/documentation/concepts/collections/#create-a-collection
class QdrantDB(VectorDB):
    def __init__(
        self,
        collections: list[dict[str, str | int]],
        recreate: bool = False,
        host: str = "localhost",
        port: int = 6333,
        api_key: str | None = None,
    ):
        super().__init__(collections)
        self.client = QdrantClient(host=host, port=port, api_key=api_key)
        if len(collections) < 1:
            raise ValueError('Provide at least 1 collection, e.g. [{"name": "my_collec", "size": 512}]')

        # TODO: add indexing for id and sequence in payload
        if recreate:
            for collection in collections:
                self.client.recreate_collection(
                    collection_name=collection["name"],
                    vectors_config=VectorParams(size=collection["size"], distance=Distance.COSINE),
                )
                self.client.create_payload_index(collection["name"], "id", "keyword")
        else:
            try:
                collec_list = [
                    f"{self.client.get_collection(collec['name']).points_count} {collec['name']}"
                    for collec in collections
                ]
                log.info(f"Vector DB initialized: {' | '.join(collec_list)}")
            except Exception as e:
                log.info(f"⚠️ Collection not found: {e}, recreating the collections")
                for collection in collections:
                    self.client.recreate_collection(
                        collection_name=collection["name"],
                        vectors_config=VectorParams(size=collection["size"], distance=Distance.COSINE),
                        # Qdrant supports Dot, Cosine and Euclid
                    )
                    self.client.create_payload_index(
                        collection["name"],
                        "id",
                        {
                            "type": "text",
                            "tokenizer": "word",
                            "min_token_len": 2,
                            "max_token_len": 30,
                            # "lowercase": True
                        },
                    )

    def add(self, collection_name: str, item_list: list[str]) -> UpdateResult:
        """Add a list of entities and their vector to the database"""
        batch_size = 1000
        for i in range(0, len(item_list), batch_size):
            item_batch = item_list[i : i + batch_size]
            # TODO: load per 1000
            points_count = self.client.get_collection(collection_name).points_count
            points_list = [
                PointStruct(id=points_count + i + 1, vector=item["vector"], payload=item["payload"])
                for i, item in enumerate(item_batch)
            ]
            # PointStruct(id=2, vector=[0.19, 0.81, 0.75, 0.11], payload={"city": "London"}),
            operation_info = self.client.upsert(
                collection_name=collection_name,
                wait=True,
                points=points_list,
            )
        return operation_info

    def get(
        self, collection_name: str, search_input: str | None = None, search_field: str = "id", limit: int = 5
    ) -> list[Any]:
        """Get the vector for a specific entity ID"""
        # if search_input and ":" in search_input:
        #     search_input = search_input.split(":", 1)[1]
        search_result = self.client.scroll(
            collection_name=collection_name,
            scroll_filter=Filter(should=[FieldCondition(key=search_field, match=MatchText(text=search_input))])
            if search_input
            else None,
            with_vectors=True,
            with_payload=True,
            limit=limit,
        )
        return search_result[0]


    def search(
        self, collection_name: str, vector: str, search_input: str | None = None, limit: int = 10
    ) -> list[Any] | None:
        """Search for vectors similar to a given vector"""
        search_result = self.client.search(
            collection_name=collection_name,
            query_vector=vector,
            query_filter=Filter(must=[FieldCondition(key="id", match=MatchText(value=search_input))])
            if search_input
            else None,
            # search_params=SearchParams(hnsw_ef=128, exact=False),
            with_vectors=True,
            with_payload=True,
            limit=limit,
        )
        # "strategy": "best_score"
        return search_result


def init_vectordb(api_key: str = "TOCHANGE", recreate: bool = False, collections: list[dict[str, str]] = COLLECTIONS):
    qdrant_url = "qdrant.137.120.31.148.nip.io"
    return QdrantDB(collections=collections, recreate=recreate, host=qdrant_url, port=443, api_key=api_key)
