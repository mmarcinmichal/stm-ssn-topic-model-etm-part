"""
ETM Topic Metrics Module

This module defines the concrete subclass `ETMTopicMetrics` of the `TopicMetrics` class. It contains
methods for extracting topics and calculating metrics specific to the Embedding Topic Model (ETM).

"""
from typing import List

from model.topic.topic_metrics import TopicMetrics
from model.topic.topic_object import TopicObject


class ETMTopicMetrics(TopicMetrics):
    """
    Concrete subclass of TopicMetrics for ETM (Embedding Topic Model) metrics calculations.

    This class implements the specific methods required to extract topics and calculate metrics
    for the Embedding Topic Model.
    """

    def __init__(self, number_of_topics, model, endpoint, word_number=10):
        TopicMetrics.__init__(self, number_of_topics, model, endpoint, word_number)

    def extract_topics_from_model(self, word_number=10) -> List[TopicObject]:
        """
        Extract topics from the underlying model.
        This method extracts topics from the underlying topic model.

        Args:
            word_number (int, optional): Number of top words to include in each extracted
                topic (default is 10).

        Returns:
            List[TopicObject]: A list of TopicObject instances representing the extracted topics.
        """
        topics: List[TopicObject] = []
        for topic_id, top_words in enumerate(self.model.get_topics(word_number)):
            topics.append(TopicObject(topic_id, top_words))
        return topics
