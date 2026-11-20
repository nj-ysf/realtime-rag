import hashlib
import json
import logging

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from qdrant_client.models import PointStruct

from ingestion.config import (
    QDRANT_URL,
    QDRANT_COLLECTION
)

from ingestion.embeddings.embedding_service import EmbeddingService

logger = logging.getLogger("ingestion.qdrant")


def parse_attributes(raw):
    """Debezium sends JSONB columns as a JSON-encoded string."""
    attributes = raw or {}
    if isinstance(attributes, str):
        attributes = json.loads(attributes)
    return attributes


def canonical_attributes(attributes):
    """Deterministic JSON text: sorted keys at every nesting level, fixed separators."""
    return json.dumps(
        attributes,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def compute_semantic_hash(name, description, brand, attributes):
    """
    Hash of exactly the fields that feed the embedding.
    Same semantic content => same hash, whatever the JSONB key order
    or whether a text field is NULL or empty.
    """
    canonical = json.dumps(
        {
            "name": name or "",
            "description": description or "",
            "brand": brand or "",
            "attributes": canonical_attributes(attributes),
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class QdrantService:

    def __init__(self):
        self.client = QdrantClient(url=QDRANT_URL)
        self.embedding_service = EmbeddingService()

        self.create_collection()

    def create_collection(self):
        collections = self.client.get_collections().collections

        exists = any(
            collection.name == QDRANT_COLLECTION
            for collection in collections
        )

        if not exists:
            dimension = self.embedding_service.model.get_sentence_embedding_dimension()

            self.client.create_collection(
                collection_name=QDRANT_COLLECTION,
                vectors_config=VectorParams(
                    size=dimension,
                    distance=Distance.COSINE
                )
            )

            logger.info("Qdrant collection created!")

        else:
            logger.info("Qdrant collection already exists!")

    def get_stored_semantic_hash(self, product_id):
        """
        Returns (point_exists, stored_hash).
        stored_hash is None when the point is new or predates semantic_hash.
        """
        points = self.client.retrieve(
            collection_name=QDRANT_COLLECTION,
            ids=[product_id],
            with_payload=["semantic_hash"],
            with_vectors=False,
        )

        if not points:
            return False, None

        return True, (points[0].payload or {}).get("semantic_hash")

    def upsert_product(self, product):
        product_id = product["id"]

        attributes = parse_attributes(product.get("attributes"))

        semantic_hash = compute_semantic_hash(
            product.get("name"),
            product.get("description"),
            product.get("brand"),
            attributes,
        )

        # Product information stored as metadata
        payload = {
            "product_id": product_id,
            "name": product.get("name"),
            "brand": product.get("brand"),
            "description": product.get("description"),
            "category_id": product.get("category_id"),
            "price": str(product.get("price")),
            "currency": product.get("currency"),
            "stock_quantity": product.get("stock_quantity"),
            "is_active": product.get("is_active"),
            "attributes": attributes,
            "semantic_hash": semantic_hash,
        }

        point_exists, stored_hash = self.get_stored_semantic_hash(product_id)

        if point_exists and stored_hash == semantic_hash:
            # Embedding-relevant fields unchanged (e.g. only price/stock changed):
            # keep the existing vector and refresh the payload only.
            self.client.set_payload(
                collection_name=QDRANT_COLLECTION,
                payload=payload,
                points=[product_id],
            )

            logger.info(
                "Embedding skipped (semantic hash unchanged), payload updated: product %s",
                product_id
            )
            return

        if not point_exists:
            reason = "new product"
        elif stored_hash is None:
            reason = "no stored semantic_hash"
        else:
            reason = "semantic fields changed"

        # Text used to generate the semantic embedding. Attributes are
        # serialized canonically so identical hash => identical text.
        document = f"""
        Product: {product.get('name') or ''}
        Brand: {product.get('brand') or ''}
        Description: {product.get('description') or ''}
        Attributes: {canonical_attributes(attributes)}
        """

        vector = self.embedding_service.generate_embedding(document)
        logger.info(
            "Embedding regenerated (%s) for product %s (dim=%s)",
            reason, product_id, len(vector)
        )

        self.client.upsert(
            collection_name=QDRANT_COLLECTION,
            points=[
                PointStruct(
                    id=product_id,
                    vector=vector,
                    payload=payload
                )
            ]
        )

        logger.info("Qdrant upsert successful: product %s", product_id)

    def delete_product(self, product_id):
        self.client.delete(
            collection_name=QDRANT_COLLECTION,
            points_selector=[product_id]
        )

        logger.info("Qdrant delete successful: product %s", product_id)
