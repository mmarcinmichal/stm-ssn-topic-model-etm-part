"""
Create retrival metrics for differ types of topic modelling methods, like a:
LDA, BTM, ETM, and STM
"""
import os
import pickle


import numpy as np

from model.evaluation.classification_metrics import ClassificationMetrics


class RetrivalMetrics:
    def __init__(
        self,
        model_name,
        topic_nbr,
        train_topic_probability,
        test_topic_probability,
        train_labels,
        test_labels,
        categories,
    ):
        self.topic_nbr = topic_nbr
        self.test_topic_probability = test_topic_probability
        self.train_topic_probability = train_topic_probability
        self.model_name = model_name
        self.categories = categories
        self.test_labels = test_labels
        self.train_labels = train_labels
        self.classification_metrics: ClassificationMetrics = None
        self.purity = None
        self.purity = 0

    def calculate_metrics(self):
        predicated_labels, test_labels = self.process_data()
        self.classification_metrics = ClassificationMetrics(
            true_y=test_labels, pred_y=predicated_labels, categories=self.categories
        )

    def process_data(self):
        self.calculate_purity()
        test_labels, train_labels = self.retrieve_docs()
        return train_labels, test_labels

    def retrieve_docs(self):
        start_idx = 0
        test_labels, train_labels = [], []
        request_size = 100
        step = 5000
        test_size = self.test_topic_probability.shape[0]
        print("IR task started")
        while start_idx < self.test_topic_probability.shape[0]:
            print(f"{start_idx} of {test_size}")
            res_matrix = np.matmul(
                self.test_topic_probability[start_idx : start_idx + step],
                self.train_topic_probability.T,
            )
            for idx, row in enumerate(res_matrix):
                docs_closeset = np.argsort(row)[-request_size:][::-1]
                labels_closests = [(id, self.train_labels[id]) for id in docs_closeset]
                document_test_class = self.test_labels[start_idx + idx]
                for id, l in labels_closests:
                    test_labels.append(document_test_class)
                    train_labels.append(l)
            start_idx += step
        return test_labels, train_labels

    def calculate_purity(self):
        cluster_label_assignment = np.zeros((self.topic_nbr, len(self.categories)))
        for idx, row in enumerate(self.test_topic_probability):
            max_prob_top = np.argmax(row)
            label = self.test_labels[idx]
            cluster_label_assignment[max_prob_top][label] += 1
        max_doc_cluster = np.sum([p for p in np.max(cluster_label_assignment, axis=1)])
        self.purity = max_doc_cluster / len(self.test_labels)

    def save(self, result_folder, model_name):
        """Save result to the given file"""
        results_path = os.path.join(result_folder, f"{model_name}_clustering_metrics")
        print(f"Saving file in {results_path}")
        with open(results_path, "wb") as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(folder, file_name):
        """Load result from the given file"""
        path = os.path.join(folder, file_name)
        with open(path, "rb") as file:
            return pickle.load(file)
