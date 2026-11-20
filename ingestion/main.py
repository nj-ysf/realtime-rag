import logging

from ingestion.consumer.kafka_consumer import KafkaProductConsumer

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )
    consumer = KafkaProductConsumer()
    consumer.start()
