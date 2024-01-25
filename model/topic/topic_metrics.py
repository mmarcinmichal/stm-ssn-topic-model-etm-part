"""
Topic Metrics Module

This module defines the abstract base class `TopicMetrics`,
along with its concrete methods and utilities
for extracting topics, calculating coherence metrics, and saving/loading metrics results.
The module also provides the ability to interact with
the Palmetto service for calculating topic coherence metrics.

"""
import os
import pickle
import time
from abc import ABC, abstractmethod
from typing import List

import requests

from model.topic.topic_object import TopicObject


class TopicMetrics(ABC):
    """
    Abstract base class for topic metrics calculation.

    This class defines the common interface and shared utilities for extracting topics,
    calculating coherence metrics, and saving/loading metrics results.
    It also provides the ability to interact with
    the Palmetto service for calculating topic coherence metrics.
    """

    def __init__(self, number_of_topics, model, endpoint, word_number):
        self.number_of_topics = number_of_topics
        self.model = model
        self.coherence_metrics = ["npmi", "ca"]
        self.endpoint = endpoint
        self.topics: List[TopicObject] = self.extract_topics_from_model(word_number)

    @abstractmethod
    def extract_topics_from_model(self, word_number=10) -> List[TopicObject]:
        """
        Abstract method to extract topics from the underlying model.

        This method must be implemented by concrete subclasses to extract topics from the underlying
        topic model. The specific implementation details will vary depending on the type of
        topic model being used.

        Args:
            word_number (int, optional): Number of top words to include in each extracted topic
                                         (default is 10).

        Returns:
            List[TopicObject]: A list of TopicObject instances representing the extracted topics.
        """

    def palmetto_request(self, metric, topic_words: List[str]):
        """
        Send a request to the Palmetto service to calculate a coherence metric for a given topic.

        This method sends an HTTP request to the Palmetto service to calculate a coherence metric for
        a given list of topic words. It constructs the URL with the provided metric and topic words,
        and waits for a successful response (HTTP status code 200). If the response is not successful,
        it retries the request after a brief delay.

        Args:
            metric (str): The coherence metric to calculate (e.g., "npmi").
            topic_words (List[str]): A list of topic words for which the metric will be calculated.

        Returns:
            float: The calculated coherence metric value for the given topic words.
        """
        response = requests.get(
            f"{self.endpoint}{metric}?words={'%20'.join(topic_words)}"
        )
        while response.status_code != 200:
            response = requests.get(
                f"{self.endpoint}{metric}?words={'%20'.join(topic_words)}"
            )
            time.sleep(5)
            print("Retrying request ...")

        return float(response.text)

    def generate_metrics(self):
        """
        Calculate and assign coherence metrics to topics.

        This method calculates coherence metrics for each topic in the collection and assigns
        the calculated metrics to each topic. It also calculates a metric based on the uniqueness
        of words in topics and prints the results for each topic.
        """
        unique = set()
        for topic in self.topics:
            unique.update(topic.words)
            print(f"Debug:: Unique words in topic: {topic.words}")

        print(
            f"Debug:: Metric based on uniques: {len(unique) / (len(self.topics) * 10)}"
        )

        for topic in self.topics:
            print(f"Requesting metrics for {topic.words}")
            for metric in self.coherence_metrics:
                metric_val = self.palmetto_request(metric, topic_words=topic.words)
                topic.set_metric(metric, metric_val)

    def save(self, folder, name):
        """Save result to the given file"""
        results_path = os.path.join(folder, name)
        with open(results_path, "wb") as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(folder, name):
        """Load result from the given file"""
        path = os.path.join(folder, name)
        with open(path, "rb") as file:
            return pickle.load(file)

    def get_average_metric_for_top_n(self, metric, number_of_top_n=100.0) -> float:
        """Generate average metric for the given number of topics"""
        topics_for_evaluation = self.get_top_n(metric, number_of_top_n)
        avg = round(
            sum(map(lambda t: t.get_metric(metric), topics_for_evaluation))
            / len(topics_for_evaluation),
            5,
        )
        return avg

    def get_top_n(self, metric: str, number_of_top_n=100.0) -> List[TopicObject]:
        """Get static of metric fot top n topics"""
        topic_number = int((number_of_top_n / 100) * self.number_of_topics)
        topics_for_extraction: List[TopicObject] = sorted(
            self.topics, key=lambda x: x.get_metric(metric), reverse=True
        )
        return topics_for_extraction[:topic_number]
