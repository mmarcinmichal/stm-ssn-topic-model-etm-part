import os
import pickle
import string

import requests
from typing import List
import time

from model.topic.topic_object import TopicObject


class TopicMetrics:

    def __init__(self, name, topics: List, result_folder='output-data/clustering'):
        self.result_folder = result_folder
        self.name = name
        self.number_of_topics = len(topics)
        self.coherence_metrics = ['ca', 'npmi', 'uci']
        self.topics: List[TopicObject] = topics

    def palmetto_request(self, metric, topic_words: [], endpoint):
        body = '{endpint}{metric}?words={tokens}'.format(endpint=endpoint, metric=metric,
                                                                 tokens='%20'.join(topic_words))
        response = requests.get(body)
        while response.status_code != 200:
            response = requests.get('{endpint}{metric}?words={tokens}'.format(endpint=endpoint, metric=metric,
                                                                              tokens='%20'.join(topic_words)))
            time.sleep(5)
            print("retrying")

        return float(response.text)

    def generate_metrics(self, endpoint):
        for topic in self.topics:
            for metric in self.coherence_metrics:
                print("Requestign metric {metric} for {topic_id}".format(metric=metric, topic_id=topic.words))
                metric_val = self.palmetto_request(metric, topic.words, endpoint)
                topic.set_metric(metric, metric_val)

    def save_metrics(self):
        results_path = os.path.join(self.result_folder,
                                    f'topic_metrics_{self.name}')
        print(f'Saving file in {results_path}')
        with open(results_path, 'wb') as outp:
            pickle.dump(self, outp, pickle.HIGHEST_PROTOCOL)

    def get_average_metric_for_top_n(self, metric, n=100.0) -> [List[TopicObject], float]:
        topics_for_evaluation = self.get_top_n(metric, n)
        avg = round(sum(map(lambda t: t.get_metric(metric), topics_for_evaluation)) / len(topics_for_evaluation), 5)
        return topics_for_evaluation, avg

    def get_top_n(self, metric: string, n=100.0) -> List[TopicObject]:
        topic_number = int((n / 100) * self.number_of_topics)
        topics_for_extraction: List[TopicObject] = sorted(self.topics, key=lambda x: x.get_metric(metric), reverse=True)
        return topics_for_extraction[:topic_number]

    def get_diversity(self):
        unique_words = set()
        for t in self.topics:
            unique_words.add(t.words)
        return len(unique_words)/(len(self.topics)*len(self.topics[0].words))


def load_topic_metrics(folder, file_name):
    path = os.path.join(folder, file_name)
    modelObject = pickle.load(open(path, "rb"))
    return modelObject
