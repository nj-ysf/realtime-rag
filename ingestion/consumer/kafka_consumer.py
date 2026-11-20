import json
import logging
import time

from confluent_kafka import Consumer, TopicPartition

from ingestion.config import (
    KAFKA_BOOTSTRAP_SERVERS,
    KAFKA_TOPIC,
    KAFKA_GROUP_ID
)

from ingestion.vector_store.qdrant_service import QdrantService

logger = logging.getLogger("ingestion.consumer")

MAX_RETRY_DELAY_SECONDS = 30


class KafkaProductConsumer:

    def __init__(self):
        self.consumer = Consumer({
            "bootstrap.servers": KAFKA_BOOTSTRAP_SERVERS,
            "group.id": KAFKA_GROUP_ID,
            "auto.offset.reset": "earliest",
            "enable.auto.commit": False
        })

        self.qdrant = QdrantService()

    def process_event(self, message):
        raw_value = message.value()

        # Debezium emits a tombstone (null value) after every DELETE event.
        # It carries no data; the DELETE itself was already handled.
        if raw_value is None:
            logger.info("Tombstone received (key=%s), nothing to do", message.key())
            return

        event = json.loads(raw_value.decode("utf-8"))

        # With value.converter.schemas.enable=false the Debezium envelope is the
        # top-level object. With schemas enabled it is wrapped in "payload".
        payload = event.get("payload", event)

        operation = payload.get("op")
        after = payload.get("after")
        before = payload.get("before")

        logger.info(
            "Event received: op=%s topic=%s partition=%s offset=%s",
            operation, message.topic(), message.partition(), message.offset()
        )

        if operation in ("c", "u", "r") and after:
            self.qdrant.upsert_product(after)

        elif operation == "d":
            # With the default REPLICA IDENTITY, "before" only holds the primary key.
            # Fall back to the Kafka message key if "before" is missing.
            if before and before.get("id"):
                product_id = before["id"]
            else:
                product_id = json.loads(message.key().decode("utf-8"))["id"]

            self.qdrant.delete_product(product_id)

        else:
            # Never drop an event silently: unknown ops / empty payloads are surfaced.
            raise ValueError(
                f"Unhandled Debezium event (op={operation!r}, "
                f"has_after={bool(after)}, has_before={bool(before)})"
            )

    def start(self):
        self.consumer.subscribe([KAFKA_TOPIC])

        logger.info(
            "Kafka consumer started (bootstrap=%s, topic=%s, group=%s)",
            KAFKA_BOOTSTRAP_SERVERS, KAFKA_TOPIC, KAFKA_GROUP_ID
        )

        failures = 0

        try:
            while True:
                message = self.consumer.poll(1.0)

                if message is None:
                    continue

                if message.error():
                    logger.error("Kafka error: %s", message.error())
                    continue

                try:
                    self.process_event(message)

                except Exception:
                    # Do not commit and do not skip: rewind to the failed message
                    # and retry it with a capped exponential backoff.
                    failures += 1
                    delay = min(2 ** failures, MAX_RETRY_DELAY_SECONDS)
                    logger.exception(
                        "Processing failed at offset %s (attempt %s), retrying in %ss",
                        message.offset(), failures, delay
                    )
                    self.consumer.seek(TopicPartition(
                        message.topic(), message.partition(), message.offset()
                    ))
                    time.sleep(delay)
                    continue

                # Commit only after the event was fully processed.
                failures = 0
                self.consumer.commit(message=message, asynchronous=False)
                logger.info("Offset %s committed", message.offset())

        except KeyboardInterrupt:
            logger.info("Consumer stopped.")

        finally:
            self.consumer.close()
