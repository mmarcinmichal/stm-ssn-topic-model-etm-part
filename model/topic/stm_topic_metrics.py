"""
STM Topic Metrics Module

This module defines the concrete subclass `STMTopicMetrics` of the `TopicMetrics` class. It contains
methods for extracting topics and calculating metrics specific to the STM.

"""
from typing import List

from model.network.stm_model_runner import STMModelRunner
from model.topic.topic_metrics import TopicMetrics
from model.topic.topic_object import TopicObject


class STMTopicMetrics(TopicMetrics):
    """
    STM Topic Metrics Module
    """

    def __init__(
        self, number_of_topics, model: STMModelRunner, endpoint, word_number=10
    ):
        super().__init__(number_of_topics, model, endpoint, word_number)

    def extract_topics_from_model(self, word_number=10) -> List[TopicObject]:
        return self.model.extract_topics_from_model(word_number)
