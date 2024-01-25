import json
import os.path
import pickle as pkl
import torch
import random
import timeit

from typing import Tuple, List, Dict, Any
from embedded_topic_model.utils import data as etm_utils_data

import nltk
import numpy as np
import pandas as pd
from embedded_topic_model.models.etm import ETM
from embedded_topic_model.utils import embedding
from embedded_topic_model.utils.preprocessing import (
    _create_dictionaries,
    _remove_empty_documents,
    _create_list_words,
    _create_document_indices,
    _create_bow,
    _split_bow,
    _to_numpy_array,
)
from plotnine import ggplot, aes, geom_point, labs, theme, element_text
from sklearn.feature_extraction.text import CountVectorizer

from model.datasets.dataset_loader import DatasetLoader
from model.evaluation.retrival_metrics import RetrivalMetrics
from model.preprocessing.dataset_interface import DatasetInterface

nltk.download("punkt")


def create_etm_datasets(
    dataset: List[str], train_size=1, min_df=1, max_df=100_000, debug_mode=False
) -> Tuple[list, dict, dict]:
    """The modification of the original dataset creator from the embedded_topic_model package.
    In this modification we stopped pre-processing of dataset.
    All pre-processing is made in model.datasets.dataset_loader.DatasetLoader
    """
    if debug_mode:
        print("Running modified create_etm_datasets...")

    vectorizer = CountVectorizer(min_df=min_df, max_df=max_df)
    vectorized_documents = vectorizer.fit_transform(dataset)

    documents_without_stop_words = [
        [word for word in document.split()] for document in dataset
    ]

    signed_documents = vectorized_documents.sign()

    if debug_mode:
        print("Building vocabulary...")

    sum_counts = signed_documents.sum(axis=0)
    v_size = sum_counts.shape[1]
    sum_counts_np = np.zeros(v_size, dtype=int)
    for v in range(v_size):
        sum_counts_np[v] = sum_counts[0, v]
    word2id = dict([(w, vectorizer.vocabulary_.get(w)) for w in vectorizer.vocabulary_])
    id2word = dict([(vectorizer.vocabulary_.get(w), w) for w in vectorizer.vocabulary_])

    if debug_mode:
        print("Initial vocabulary size: {}".format(v_size))

    # Sort elements in vocabulary
    idx_sort = np.argsort(sum_counts_np)

    # Creates vocabulary
    vocabulary = [id2word[idx_sort[cc]] for cc in range(v_size)]

    if debug_mode:
        print("Tokenizing documents and splitting into train/test...")

    num_docs = signed_documents.shape[0]
    train_dataset_size = train_size
    test_dataset_size = int(num_docs - train_dataset_size)
    idx_permute = np.arange(num_docs).astype(int)

    if debug_mode:
        print(
            f"train_dataset_size {train_dataset_size} test_dataset_size {test_dataset_size} sum {train_dataset_size + test_dataset_size}"
        )

    # Remove words not in train_data
    vocabulary = list(
        set(
            [
                w
                for idx_d in range(train_dataset_size)
                for w in documents_without_stop_words[idx_permute[idx_d]]
                if w in word2id
            ]
        )
    )

    # Create dictionary and inverse dictionary
    word2id, id2word = _create_dictionaries(vocabulary)

    if debug_mode:
        print(
            "vocabulary after removing words not in train: {}".format(len(vocabulary))
        )

    docs_train = [
        [
            word2id[w]
            for w in documents_without_stop_words[idx_permute[idx_d]]
            if w in word2id
        ]
        for idx_d in range(train_dataset_size)
    ]
    docs_test = [
        [
            word2id[w]
            for w in documents_without_stop_words[
                idx_permute[idx_d + train_dataset_size]
            ]
            if w in word2id
        ]
        for idx_d in range(test_dataset_size)
    ]

    if debug_mode:
        print(
            "Number of documents (train_dataset): {} [this should be equal to {}]".format(
                len(docs_train), train_dataset_size
            )
        )
        print(
            "Number of documents (test_dataset): {} [this should be equal to {}]".format(
                len(docs_test), test_dataset_size
            )
        )

    if debug_mode:
        print("Start removing empty documents...")
        print(f"docs_train: {len(docs_train)}")
        print(f"docs_train: {len(docs_test)}")

    docs_train = _remove_empty_documents(docs_train)
    docs_test = _remove_empty_documents(docs_test)

    if debug_mode:
        print("End removing empty documents...")
        print(f"docs_train: {len(docs_train)}")
        print(f"docs_train: {len(docs_test)}")

    if debug_mode:
        print("Remove test documents with length=1")
        print(f"docs_test: {len(docs_test)}")

    # docs_test = [doc for doc in docs_test if len(doc) > 1]
    # Remove test documents with length=1
    for index, doc in enumerate(docs_test):
        if len(doc) == 1:
            print("Add random words from vocabulary to fill requirements len(doc) > 1")
            print(f"Document before filling {docs_test[index]}")
            random_word = random.choice(list(vocabulary))
            id_random_word = word2id[random_word]
            docs_test[index] = doc + [id_random_word]
            print(f"Document after filling {docs_test[index]}")

    # Obtains the training and test datasets as word lists
    words_train = [[id2word[w] for w in doc] for doc in docs_train]
    words_test = [[id2word[w] for w in doc] for doc in docs_test]

    docs_test_h1 = [
        [w for i, w in enumerate(doc) if i <= len(doc) / 2.0 - 1] for doc in docs_test
    ]
    docs_test_h2 = [
        [w for i, w in enumerate(doc) if i > len(doc) / 2.0 - 1] for doc in docs_test
    ]

    words_train = _create_list_words(docs_train)
    words_test = _create_list_words(docs_test)
    words_ts_h1 = _create_list_words(docs_test_h1)
    words_ts_h2 = _create_list_words(docs_test_h2)

    if debug_mode:
        print("len(words_train): ", len(words_train))
        print("len(words_test): ", len(words_test))
        print("len(words_ts_h1): ", len(words_ts_h1))
        print("len(words_ts_h2): ", len(words_ts_h2))

    doc_indices_train = _create_document_indices(docs_train)
    doc_indices_test = _create_document_indices(docs_test)
    doc_indices_test_h1 = _create_document_indices(docs_test_h1)
    doc_indices_test_h2 = _create_document_indices(docs_test_h2)

    if debug_mode:
        print(
            "len(np.unique(doc_indices_train)): {} [this should be {}]".format(
                len(np.unique(doc_indices_train)), len(docs_train)
            )
        )
        print(
            "len(np.unique(doc_indices_test)): {} [this should be {}]".format(
                len(np.unique(doc_indices_test)), len(docs_test)
            )
        )
        print(
            "len(np.unique(doc_indices_test_h1)): {} [this should be {}]".format(
                len(np.unique(doc_indices_test_h1)), len(docs_test_h1)
            )
        )
        print(
            "len(np.unique(doc_indices_test_h2)): {} [this should be {}]".format(
                len(np.unique(doc_indices_test_h2)), len(docs_test_h2)
            )
        )

    # Number of documents in each set
    n_docs_train = len(docs_train)
    n_docs_test = len(docs_test)
    n_docs_test_h1 = len(docs_test_h1)
    n_docs_test_h2 = len(docs_test_h2)

    bow_train = _create_bow(
        doc_indices_train, words_train, n_docs_train, len(vocabulary)
    )
    bow_test = _create_bow(doc_indices_test, words_test, n_docs_test, len(vocabulary))
    bow_test_h1 = _create_bow(
        doc_indices_test_h1, words_ts_h1, n_docs_test_h1, len(vocabulary)
    )
    bow_test_h2 = _create_bow(
        doc_indices_test_h2, words_ts_h2, n_docs_test_h2, len(vocabulary)
    )

    bow_train_tokens, bow_train_counts = _split_bow(bow_train, n_docs_train)
    bow_test_tokens, bow_test_counts = _split_bow(bow_test, n_docs_test)

    bow_test_h1_tokens, bow_test_h1_counts = _split_bow(bow_test_h1, n_docs_test_h1)
    bow_test_h2_tokens, bow_test_h2_counts = _split_bow(bow_test_h2, n_docs_test_h2)

    train_dataset = {
        "tokens": _to_numpy_array(bow_train_tokens),
        "counts": _to_numpy_array(bow_train_counts),
    }

    test_dataset = {
        "test": {
            "tokens": _to_numpy_array(bow_test_tokens),
            "counts": _to_numpy_array(bow_test_counts),
        },
        "test1": {
            "tokens": _to_numpy_array(bow_test_h1_tokens),
            "counts": _to_numpy_array(bow_test_h1_counts),
        },
        "test2": {
            "tokens": _to_numpy_array(bow_test_h2_tokens),
            "counts": _to_numpy_array(bow_test_h2_counts),
        },
    }

    return vocabulary, train_dataset, test_dataset


def get_test_document_topic_dist(etm_instance: ETM, test_dataset):
    """
    Get the document-topic distribution for the test dataset using an ETM instance.

    Args:
        etm_instance (ETM): An instance of the ETM class.
        test_dataset: The test dataset to compute the document-topic distribution.

    Returns:
        torch.Tensor: The concatenated document-topic distribution for the test dataset.
    """
    etm_instance._set_test_data(test_dataset)
    etm_instance.model.eval()

    with torch.no_grad():
        indices = torch.split(
            torch.tensor(range(etm_instance.num_docs_test)),
            etm_instance.num_docs_test,
        )

        thetas = []

        for _, ind in enumerate(indices):
            print(f"indices: {ind}")
            data_batch = etm_utils_data.get_batch(
                etm_instance.test_tokens,
                etm_instance.test_counts,
                ind,
                etm_instance.vocabulary_size,
                etm_instance.device,
            )
            _sums = data_batch.sum(1).unsqueeze(1)
            normalized_data_batch = (
                data_batch / _sums if etm_instance.bow_norm else data_batch
            )
            theta, _ = etm_instance.model.get_theta(normalized_data_batch)

            thetas.append(theta)

        return torch.cat(tuple(thetas), 0)


# Currently we have default datasets such as: 20news, ag, and bbc
DATA_SET = "ag"
# For: 20news, and ag datasets set 5_000, and for bbc set 2_000
FEATURE_LIMIT = 5_000
STEPS_RANGE_NUMBER = 5

DATA_SET_PATH = f"model-input-data/{DATA_SET}"
OUTPUT_PATH = f"model-output-data/{DATA_SET}"

# Tested configuration for 20 newsgroups dataset (2000 features) - systematic grid search
configurations_20news_2000: dict = {
    # No randomized trials i.e. fixed seeds to take best results, i.e. models on training set (
    # f-score, purity, npmi) to next deeper fine-tuning (parametrization)
    # "etm_1": {
    #     "num_topics": 20,
    #     "epochs": 300,
    #     "train_embeddings": False,
    #     "isTestOnTrainingSet": True,
    #     "seeds": [2019, 2019, 2019, 2019, 2019],
    # },
    # "etm_2": {
    #     "num_topics": 20,
    #     "epochs": 600,
    #     "train_embeddings": False,
    #     "isTestOnTrainingSet": True,
    #     "seeds": [2019, 2019, 2019, 2019, 2019],
    # },
    # "etm_3": {
    #     "num_topics": 20,
    #     "epochs": 900,
    #     "train_embeddings": False,
    #     "isTestOnTrainingSet": True,
    #     "seeds": [2019, 2019, 2019, 2019, 2019],
    # },
    # "etm_4": {
    #     "num_topics": 20,
    #     "epochs": 1_500,
    #     "train_embeddings": False,
    #     "isTestOnTrainingSet": True,
    #     "seeds": [2019, 2019, 2019, 2019, 2019],
    # },
    # "etm_5": {
    #     "num_topics": 20,
    #     "epochs": 1_500,
    #     "train_embeddings": False,
    #     "isTestOnTrainingSet": True,
    #     "seeds": [2019, 2019, 2019, 2019, 2019],
    # },
    # "etm_6": {
    #     "num_topics": 20,
    #     "epochs": 300,
    #     "train_embeddings": True,
    #     "isTestOnTrainingSet": True,
    #     "seeds": [2019, 2019, 2019, 2019, 2019],
    # },
    # "etm_7": {
    #     "num_topics": 20,
    #     "epochs": 600,
    #     "train_embeddings": True,
    #     "isTestOnTrainingSet": True,
    #     "seeds": [2019, 2019, 2019, 2019, 2019],
    # },
    # "etm_8": {
    #     "num_topics": 20,
    #     "epochs": 900,
    #     "train_embeddings": True,
    #     "isTestOnTrainingSet": True,
    #     "seeds": [2019, 2019, 2019, 2019, 2019],
    # },
    # "etm_9": {
    #     "num_topics": 20,
    #     "epochs": 1_500,
    #     "train_embeddings": True,
    #     "isTestOnTrainingSet": True,
    #     "seeds": [2019, 2019, 2019, 2019, 2019],
    # },
    # "etm_10": {
    #     "num_topics": 20,
    #     "epochs": 1_500,
    #     "train_embeddings": True,
    #     "isTestOnTrainingSet": True,
    #     "seeds": [2019, 2019, 2019, 2019, 2019],
    # },
    # "etm_11": {
    #     "num_topics": 30,
    #     "epochs": 300,
    #     "train_embeddings": False,
    #     "isTestOnTrainingSet": True,
    #     "seeds": [2019, 2019, 2019, 2019, 2019],
    # },
    # "etm_12": {
    #     "num_topics": 30,
    #     "epochs": 600,
    #     "train_embeddings": False,
    #     "isTestOnTrainingSet": True,
    #     "seeds": [2019, 2019, 2019, 2019, 2019],
    # },
    # "etm_13": {
    #     "num_topics": 30,
    #     "epochs": 900,
    #     "train_embeddings": False,
    #     "isTestOnTrainingSet": True,
    #     "seeds": [2019, 2019, 2019, 2019, 2019],
    # },
    # "etm_14": {
    #     "num_topics": 30,
    #     "epochs": 1_500,
    #     "train_embeddings": False,
    #     "isTestOnTrainingSet": True,
    #     "seeds": [2019, 2019, 2019, 2019, 2019],
    # },
    # "etm_15": {
    #     "num_topics": 30,
    #     "epochs": 1_500,
    #     "train_embeddings": False,
    #     "isTestOnTrainingSet": True,
    #     "seeds": [2019, 2019, 2019, 2019, 2019],
    # },
    # "etm_16": {
    #     "num_topics": 30,
    #     "epochs": 300,
    #     "train_embeddings": True,
    #     "isTestOnTrainingSet": True,
    #     "seeds": [2019, 2019, 2019, 2019, 2019],
    # },
    # "etm_17": {
    #     "num_topics": 30,
    #     "epochs": 600,
    #     "train_embeddings": True,
    #     "isTestOnTrainingSet": True,
    #     "seeds": [2019, 2019, 2019, 2019, 2019],
    # },
    # "etm_18": {
    #     "num_topics": 30,
    #     "epochs": 900,
    #     "train_embeddings": True,
    #     "isTestOnTrainingSet": True,
    #     "seeds": [2019, 2019, 2019, 2019, 2019],
    # },
    # "etm_19": {
    #     "num_topics": 30,
    #     "epochs": 1_500,
    #     "train_embeddings": True,
    #     "isTestOnTrainingSet": True,
    #     "seeds": [2019, 2019, 2019, 2019, 2019],
    # },
    # "etm_20": {
    #     "num_topics": 30,
    #     "epochs": 1_500,
    #     "train_embeddings": True,
    #     "isTestOnTrainingSet": True,
    #     "seeds": [2019, 2019, 2019, 2019, 2019],
    # },
    # "etm_21": {
    #     "num_topics": 40,
    #     "epochs": 300,
    #     "train_embeddings": False,
    #     "isTestOnTrainingSet": True,
    #     "seeds": [2019, 2019, 2019, 2019, 2019],
    # },
    # "etm_22": {
    #     "num_topics": 40,
    #     "epochs": 600,
    #     "train_embeddings": False,
    #     "isTestOnTrainingSet": True,
    #     "seeds": [2019, 2019, 2019, 2019, 2019],
    # },
    # "etm_23": {
    #     "num_topics": 40,
    #     "epochs": 900,
    #     "train_embeddings": False,
    #     "isTestOnTrainingSet": True,
    #     "seeds": [2019, 2019, 2019, 2019, 2019],
    # },
    # "etm_24": {
    #     "num_topics": 40,
    #     "epochs": 1_500,
    #     "train_embeddings": False,
    #     "isTestOnTrainingSet": True,
    #     "seeds": [2019, 2019, 2019, 2019, 2019],
    # },
    # "etm_25": {
    #     "num_topics": 40,
    #     "epochs": 1_500,
    #     "train_embeddings": False,
    #     "isTestOnTrainingSet": True,
    #     "seeds": [2019, 2019, 2019, 2019, 2019],
    # },
    # "etm_26": {
    #     "num_topics": 40,
    #     "epochs": 300,
    #     "train_embeddings": True,
    #     "isTestOnTrainingSet": True,
    #     "seeds": [2019, 2019, 2019, 2019, 2019],
    # },
    # "etm_27": {
    #     "num_topics": 40,
    #     "epochs": 600,
    #     "train_embeddings": True,
    #     "isTestOnTrainingSet": True,
    #     "seeds": [2019, 2019, 2019, 2019, 2019],
    # },
    # "etm_28": {
    #     "num_topics": 40,
    #     "epochs": 900,
    #     "train_embeddings": True,
    #     "isTestOnTrainingSet": True,
    #     "seeds": [2019, 2019, 2019, 2019, 2019],
    # },
    # "etm_29": {
    #     "num_topics": 40,
    #     "epochs": 1_500,
    #     "train_embeddings": True,
    #     "isTestOnTrainingSet": True,
    #     "seeds": [2019, 2019, 2019, 2019, 2019],
    # },
    # "etm_30": {
    #     "num_topics": 40,
    #     "epochs": 1_500,
    #     "train_embeddings": True,
    #     "isTestOnTrainingSet": True,
    #     "seeds": [2019, 2019, 2019, 2019, 2019],
    # },
    # Randomized trials for best results on training set (f-score, purity, npmi)
    # "etm_31": {
    #     "num_topics": 20,
    #     "epochs": 900,
    #     "train_embeddings": False,
    #     "isTestOnTrainingSet": True,
    #     "seeds": [2019, 1, 7, 28, 517],
    # },
    # "etm_32": {
    #     "num_topics": 20,
    #     "epochs": 900,
    #     "train_embeddings": True,
    #     "isTestOnTrainingSet": True,
    #     "seeds": [2019, 1, 7, 28, 517],
    # },
    # "etm_33": {
    #     "num_topics": 30,
    #     "epochs": 900,
    #     "train_embeddings": False,
    #     "isTestOnTrainingSet": True,
    #     "seeds": [2019, 1, 7, 28, 517],
    # },
    # "etm_34": {
    #     "num_topics": 30,
    #     "epochs": 900,
    #     "train_embeddings": True,
    #     "isTestOnTrainingSet": True,
    #     "seeds": [2019, 1, 7, 28, 517],
    # },
    # "etm_35": {
    #     "num_topics": 40,
    #     "epochs": 300,
    #     "train_embeddings": False,
    #     "isTestOnTrainingSet": True,
    #     "seeds": [2019, 1, 7, 28, 517],
    # },
    # "etm_36": {
    #     "num_topics": 40,
    #     "epochs": 300,
    #     "train_embeddings": True,
    #     "isTestOnTrainingSet": True,
    #     "seeds": [2019, 1, 7, 28, 517],
    # },
    # "etm_37": {
    #     "num_topics": 40,
    #     "epochs": 600,
    #     "train_embeddings": False,
    #     "isTestOnTrainingSet": True,
    #     "seeds": [2019, 1, 7, 28, 517],
    # },
    # "etm_38": {
    #     "num_topics": 40,
    #     "epochs": 600,
    #     "train_embeddings": True,
    #     "isTestOnTrainingSet": True,
    #     "seeds": [2019, 1, 7, 28, 517],
    # },
    # Randomized trials for best results on test set (f-score, purity, npmi)
    # "etm_39": {
    #     "num_topics": 20,
    #     "epochs": 900,
    #     "train_embeddings": False,
    #     "isTestOnTrainingSet": False,
    #     "seeds": [2019, 1, 7, 28, 517],
    # },
    # "etm_40": {
    #     "num_topics": 20,
    #     "epochs": 900,
    #     "train_embeddings": True,
    #     "isTestOnTrainingSet": False,
    #     "seeds": [2019, 1, 7, 28, 517],
    # },
    # "etm_41": {
    #     "num_topics": 30,
    #     "epochs": 900,
    #     "train_embeddings": False,
    #     "isTestOnTrainingSet": False,
    #     "seeds": [2019, 1, 7, 28, 517],
    # },
    # "etm_42": {
    #     "num_topics": 30,
    #     "epochs": 900,
    #     "train_embeddings": True,
    #     "isTestOnTrainingSet": False,
    #     "seeds": [2019, 1, 7, 28, 517],
    # },
    # "etm_43": {
    #     "num_topics": 40,
    #     "epochs": 300,
    #     "train_embeddings": False,
    #     "isTestOnTrainingSet": False,
    #     "seeds": [2019, 1, 7, 28, 517],
    # },
    # "etm_44": {
    #     "num_topics": 40,
    #     "epochs": 300,
    #     "train_embeddings": True,
    #     "isTestOnTrainingSet": False,
    #     "seeds": [2019, 1, 7, 28, 517],
    # },
    # "etm_45": {
    #     "num_topics": 40,
    #     "epochs": 600,
    #     "train_embeddings": False,
    #     "isTestOnTrainingSet": False,
    #     "seeds": [2019, 1, 7, 28, 517],
    # },
    # "etm_46": {
    #     "num_topics": 40,
    #     "epochs": 600,
    #     "train_embeddings": True,
    #     "isTestOnTrainingSet": False,
    #     "seeds": [2019, 1, 7, 28, 517],
    # },
}

# Tested configuration for 20 newsgroups dataset (5000 features) - systematic grid search
configurations_20news: dict = {
    # No randomized trials i.e. fixed seeds to take best results, i.e. models on training set (
    # f-score, purity, npmi) to next deeper fine-tuning (parametrization)
    # "etm_1": {
    #     "num_topics": 20,
    #     "epochs": 300,
    #     "train_embeddings": False,
    #     "isTestOnTrainingSet": True,
    #     "seeds": [2019, 2019, 2019, 2019, 2019],
    # },
    # "etm_2": {
    #     "num_topics": 20,
    #     "epochs": 600,
    #     "train_embeddings": False,
    #     "isTestOnTrainingSet": True,
    #     "seeds": [2019, 2019, 2019, 2019, 2019],
    # },
    # "etm_3": {
    #     "num_topics": 20,
    #     "epochs": 900,
    #     "train_embeddings": False,
    #     "isTestOnTrainingSet": True,
    #     "seeds": [2019, 2019, 2019, 2019, 2019],
    # },
    # "etm_4": {
    #     "num_topics": 20,
    #     "epochs": 1_500,
    #     "train_embeddings": False,
    #     "isTestOnTrainingSet": True,
    #     "seeds": [2019, 2019, 2019, 2019, 2019],
    # },
    # "etm_5": {
    #     "num_topics": 20,
    #     "epochs": 1_500,
    #     "train_embeddings": False,
    #     "isTestOnTrainingSet": True,
    #     "seeds": [2019, 2019, 2019, 2019, 2019],
    # },
    # "etm_6": {
    #     "num_topics": 20,
    #     "epochs": 300,
    #     "train_embeddings": True,
    #     "isTestOnTrainingSet": True,
    #     "seeds": [2019, 2019, 2019, 2019, 2019],
    # },
    # "etm_7": {
    #     "num_topics": 20,
    #     "epochs": 600,
    #     "train_embeddings": True,
    #     "isTestOnTrainingSet": True,
    #     "seeds": [2019, 2019, 2019, 2019, 2019],
    # },
    # "etm_8": {
    #     "num_topics": 20,
    #     "epochs": 900,
    #     "train_embeddings": True,
    #     "isTestOnTrainingSet": True,
    #     "seeds": [2019, 2019, 2019, 2019, 2019],
    # },
    # "etm_9": {
    #     "num_topics": 20,
    #     "epochs": 1_500,
    #     "train_embeddings": True,
    #     "isTestOnTrainingSet": True,
    #     "seeds": [2019, 2019, 2019, 2019, 2019],
    # },
    # "etm_10": {
    #     "num_topics": 20,
    #     "epochs": 1_500,
    #     "train_embeddings": True,
    #     "isTestOnTrainingSet": True,
    #     "seeds": [2019, 2019, 2019, 2019, 2019],
    # },
    # "etm_11": {
    #     "num_topics": 30,
    #     "epochs": 300,
    #     "train_embeddings": False,
    #     "isTestOnTrainingSet": True,
    #     "seeds": [2019, 2019, 2019, 2019, 2019],
    # },
    # "etm_12": {
    #     "num_topics": 30,
    #     "epochs": 600,
    #     "train_embeddings": False,
    #     "isTestOnTrainingSet": True,
    #     "seeds": [2019, 2019, 2019, 2019, 2019],
    # },
    # "etm_13": {
    #     "num_topics": 30,
    #     "epochs": 900,
    #     "train_embeddings": False,
    #     "isTestOnTrainingSet": True,
    #     "seeds": [2019, 2019, 2019, 2019, 2019],
    # },
    # "etm_14": {
    #     "num_topics": 30,
    #     "epochs": 1_500,
    #     "train_embeddings": False,
    #     "isTestOnTrainingSet": True,
    #     "seeds": [2019, 2019, 2019, 2019, 2019],
    # },
    # "etm_15": {
    #     "num_topics": 30,
    #     "epochs": 1_500,
    #     "train_embeddings": False,
    #     "isTestOnTrainingSet": True,
    #     "seeds": [2019, 2019, 2019, 2019, 2019],
    # },
    # "etm_16": {
    #     "num_topics": 30,
    #     "epochs": 300,
    #     "train_embeddings": True,
    #     "isTestOnTrainingSet": True,
    #     "seeds": [2019, 2019, 2019, 2019, 2019],
    # },
    # "etm_17": {
    #     "num_topics": 30,
    #     "epochs": 600,
    #     "train_embeddings": True,
    #     "isTestOnTrainingSet": True,
    #     "seeds": [2019, 2019, 2019, 2019, 2019],
    # },
    # "etm_18": {
    #     "num_topics": 30,
    #     "epochs": 900,
    #     "train_embeddings": True,
    #     "isTestOnTrainingSet": True,
    #     "seeds": [2019, 2019, 2019, 2019, 2019],
    # },
    # "etm_19": {
    #     "num_topics": 30,
    #     "epochs": 1_500,
    #     "train_embeddings": True,
    #     "isTestOnTrainingSet": True,
    #     "seeds": [2019, 2019, 2019, 2019, 2019],
    # },
    # "etm_20": {
    #     "num_topics": 30,
    #     "epochs": 1_500,
    #     "train_embeddings": True,
    #     "isTestOnTrainingSet": True,
    #     "seeds": [2019, 2019, 2019, 2019, 2019],
    # },
    # "etm_21": {
    #     "num_topics": 40,
    #     "epochs": 300,
    #     "train_embeddings": False,
    #     "isTestOnTrainingSet": True,
    #     "seeds": [2019, 2019, 2019, 2019, 2019],
    # },
    # "etm_22": {
    #     "num_topics": 40,
    #     "epochs": 600,
    #     "train_embeddings": False,
    #     "isTestOnTrainingSet": True,
    #     "seeds": [2019, 2019, 2019, 2019, 2019],
    # },
    # "etm_23": {
    #     "num_topics": 40,
    #     "epochs": 900,
    #     "train_embeddings": False,
    #     "isTestOnTrainingSet": True,
    #     "seeds": [2019, 2019, 2019, 2019, 2019],
    # },
    # "etm_24": {
    #     "num_topics": 40,
    #     "epochs": 1_500,
    #     "train_embeddings": False,
    #     "isTestOnTrainingSet": True,
    #     "seeds": [2019, 2019, 2019, 2019, 2019],
    # },
    # "etm_25": {
    #     "num_topics": 40,
    #     "epochs": 1_500,
    #     "train_embeddings": False,
    #     "isTestOnTrainingSet": True,
    #     "seeds": [2019, 2019, 2019, 2019, 2019],
    # },
    # "etm_26": {
    #     "num_topics": 40,
    #     "epochs": 300,
    #     "train_embeddings": True,
    #     "isTestOnTrainingSet": True,
    #     "seeds": [2019, 2019, 2019, 2019, 2019],
    # },
    # "etm_27": {
    #     "num_topics": 40,
    #     "epochs": 600,
    #     "train_embeddings": True,
    #     "isTestOnTrainingSet": True,
    #     "seeds": [2019, 2019, 2019, 2019, 2019],
    # },
    # "etm_28": {
    #     "num_topics": 40,
    #     "epochs": 900,
    #     "train_embeddings": True,
    #     "isTestOnTrainingSet": True,
    #     "seeds": [2019, 2019, 2019, 2019, 2019],
    # },
    # "etm_29": {
    #     "num_topics": 40,
    #     "epochs": 1_500,
    #     "train_embeddings": True,
    #     "isTestOnTrainingSet": True,
    #     "seeds": [2019, 2019, 2019, 2019, 2019],
    # },
    # "etm_30": {
    #     "num_topics": 40,
    #     "epochs": 1_500,
    #     "train_embeddings": True,
    #     "isTestOnTrainingSet": True,
    #     "seeds": [2019, 2019, 2019, 2019, 2019],
    # },
    # Randomized trials for best results on training set (f-score, purity, npmi, uniques)
    # "etm_31": {
    #     "num_topics": 30,
    #     "epochs": 1_500,
    #     "train_embeddings": True,
    #     "isTestOnTrainingSet": True,
    #     "seeds": [2019, 1, 7, 28, 517],
    # },
    # "etm_32": {
    #     "num_topics": 20,
    #     "epochs": 900,
    #     "train_embeddings": False,
    #     "isTestOnTrainingSet": True,
    #     "seeds": [2019, 1, 7, 28, 517],
    # },
    # "etm_33": {
    #     "num_topics": 30,
    #     "epochs": 1_500,
    #     "train_embeddings": False,
    #     "isTestOnTrainingSet": True,
    #     "seeds": [2019, 1, 7, 28, 517],
    # },
    # "etm_34": {
    #     "num_topics": 40,
    #     "epochs": 1_500,
    #     "train_embeddings": False,
    #     "isTestOnTrainingSet": True,
    #     "seeds": [2019, 1, 7, 28, 517],
    # },
    # "etm_35": {
    #     "num_topics": 20,
    #     "epochs": 300,
    #     "train_embeddings": False,
    #     "isTestOnTrainingSet": True,
    #     "seeds": [2019, 1, 7, 28, 517],
    # },
    # "etm_36": {
    #     "num_topics": 40,
    #     "epochs": 900,
    #     "train_embeddings": False,
    #     "isTestOnTrainingSet": True,
    #     "seeds": [2019, 1, 7, 28, 517],
    # },
    # Randomized trials for best results on test set (f-score, purity, npmi, uniques)
    # "etm_37": {
    #     "num_topics": 30,
    #     "epochs": 1_500,
    #     "train_embeddings": True,
    #     "isTestOnTrainingSet": False,
    #     "seeds": [2019, 1, 7, 28, 517],
    # },
    # "etm_38": {
    #     "num_topics": 20,
    #     "epochs": 900,
    #     "train_embeddings": False,
    #     "isTestOnTrainingSet": False,
    #     "seeds": [2019, 1, 7, 28, 517],
    # },
    # "etm_39": {
    #     "num_topics": 30,
    #     "epochs": 1_500,
    #     "train_embeddings": False,
    #     "isTestOnTrainingSet": False,
    #     "seeds": [2019, 1, 7, 28, 517],
    # },
    # "etm_40": {
    #     "num_topics": 40,
    #     "epochs": 1_500,
    #     "train_embeddings": False,
    #     "isTestOnTrainingSet": False,
    #     "seeds": [2019, 1, 7, 28, 517],
    # },
    # "etm_41": {
    #     "num_topics": 20,
    #     "epochs": 300,
    #     "train_embeddings": False,
    #     "isTestOnTrainingSet": False,
    #     "seeds": [2019, 1, 7, 28, 517],
    # },
    # "etm_42": {
    #     "num_topics": 40,
    #     "epochs": 900,
    #     "train_embeddings": False,
    #     "isTestOnTrainingSet": False,
    #     "seeds": [2019, 1, 7, 28, 517],
    # },
    # Randomized trials for best results on full set (f-score, purity, npmi, uniques)
    # "etm_43": {
    #     "num_topics": 20,
    #     "epochs": 900,
    #     "train_embeddings": False,
    #     "isTestOnTrainingSet": True,
    #     "seeds": [2019, 1, 7, 28, 517],
    # },
    # "etm_44": {
    #     "num_topics": 30,
    #     "epochs": 1_500,
    #     "train_embeddings": False,
    #     "isTestOnTrainingSet": True,
    #     "seeds": [2019, 1, 7, 28, 517],
    # },
    "etm_45": {
        "num_topics": 40,
        "epochs": 900,
        "train_embeddings": False,
        "isTestOnTrainingSet": True,
        "seeds": [2019, 1, 7, 28, 517],
    },
    # Configuration of models to measure training time
    "etm_46": {
        "num_topics": 40,
        "epochs": 600,
        "train_embeddings": False,
        "isTestOnTrainingSet": True,
        "seeds": [2019, 1, 7, 28, 517],
    },
    "etm_47": {
        "num_topics": 40,
        "epochs": 300,
        "train_embeddings": False,
        "isTestOnTrainingSet": True,
        "seeds": [2019, 1, 7, 28, 517],
    },
}

# Tested configuration for bbc dataset (2000 features) - systematic grid search
configurations_bbc: dict = {
    # No randomized trials i.e. fixed seeds to take best results, i.e. models on training set (
    # f-score, purity, npmi, uniques) to next deeper fine-tuning (parametrization)
    # "etm_1": {
    #     "num_topics": 20,
    #     "epochs": 300,
    #     "train_embeddings": False,
    #     "isTestOnTrainingSet": True,
    #     "seeds": [2019, 2019, 2019, 2019, 2019],
    # },
    # "etm_2": {
    #     "num_topics": 20,
    #     "epochs": 600,
    #     "train_embeddings": False,
    #     "isTestOnTrainingSet": True,
    #     "seeds": [2019, 2019, 2019, 2019, 2019],
    # },
    # "etm_3": {
    #     "num_topics": 20,
    #     "epochs": 900,
    #     "train_embeddings": False,
    #     "isTestOnTrainingSet": True,
    #     "seeds": [2019, 2019, 2019, 2019, 2019],
    # },
    # "etm_4": {
    #     "num_topics": 20,
    #     "epochs": 1_500,
    #     "train_embeddings": False,
    #     "isTestOnTrainingSet": True,
    #     "seeds": [2019, 2019, 2019, 2019, 2019],
    # },
    # "etm_5": {
    #     "num_topics": 20,
    #     "epochs": 1_500,
    #     "train_embeddings": False,
    #     "isTestOnTrainingSet": True,
    #     "seeds": [2019, 2019, 2019, 2019, 2019],
    # },
    # "etm_6": {
    #     "num_topics": 20,
    #     "epochs": 300,
    #     "train_embeddings": True,
    #     "isTestOnTrainingSet": True,
    #     "seeds": [2019, 2019, 2019, 2019, 2019],
    # },
    # "etm_7": {
    #     "num_topics": 20,
    #     "epochs": 600,
    #     "train_embeddings": True,
    #     "isTestOnTrainingSet": True,
    #     "seeds": [2019, 2019, 2019, 2019, 2019],
    # },
    # "etm_8": {
    #     "num_topics": 20,
    #     "epochs": 900,
    #     "train_embeddings": True,
    #     "isTestOnTrainingSet": True,
    #     "seeds": [2019, 2019, 2019, 2019, 2019],
    # },
    # "etm_9": {
    #     "num_topics": 20,
    #     "epochs": 1_500,
    #     "train_embeddings": True,
    #     "isTestOnTrainingSet": True,
    #     "seeds": [2019, 2019, 2019, 2019, 2019],
    # },
    # "etm_10": {
    #     "num_topics": 20,
    #     "epochs": 1_500,
    #     "train_embeddings": True,
    #     "isTestOnTrainingSet": True,
    #     "seeds": [2019, 2019, 2019, 2019, 2019],
    # },
    # "etm_11": {
    #     "num_topics": 30,
    #     "epochs": 300,
    #     "train_embeddings": False,
    #     "isTestOnTrainingSet": True,
    #     "seeds": [2019, 2019, 2019, 2019, 2019],
    # },
    # "etm_12": {
    #     "num_topics": 30,
    #     "epochs": 600,
    #     "train_embeddings": False,
    #     "isTestOnTrainingSet": True,
    #     "seeds": [2019, 2019, 2019, 2019, 2019],
    # },
    # "etm_13": {
    #     "num_topics": 30,
    #     "epochs": 900,
    #     "train_embeddings": False,
    #     "isTestOnTrainingSet": True,
    #     "seeds": [2019, 2019, 2019, 2019, 2019],
    # },
    # "etm_14": {
    #     "num_topics": 30,
    #     "epochs": 1_500,
    #     "train_embeddings": False,
    #     "isTestOnTrainingSet": True,
    #     "seeds": [2019, 2019, 2019, 2019, 2019],
    # },
    # "etm_15": {
    #     "num_topics": 30,
    #     "epochs": 1_500,
    #     "train_embeddings": False,
    #     "isTestOnTrainingSet": True,
    #     "seeds": [2019, 2019, 2019, 2019, 2019],
    # },
    # "etm_16": {
    #     "num_topics": 30,
    #     "epochs": 300,
    #     "train_embeddings": True,
    #     "isTestOnTrainingSet": True,
    #     "seeds": [2019, 2019, 2019, 2019, 2019],
    # },
    # "etm_17": {
    #     "num_topics": 30,
    #     "epochs": 600,
    #     "train_embeddings": True,
    #     "isTestOnTrainingSet": True,
    #     "seeds": [2019, 2019, 2019, 2019, 2019],
    # },
    # "etm_18": {
    #     "num_topics": 30,
    #     "epochs": 900,
    #     "train_embeddings": True,
    #     "isTestOnTrainingSet": True,
    #     "seeds": [2019, 2019, 2019, 2019, 2019],
    # },
    # "etm_19": {
    #     "num_topics": 30,
    #     "epochs": 1_500,
    #     "train_embeddings": True,
    #     "isTestOnTrainingSet": True,
    #     "seeds": [2019, 2019, 2019, 2019, 2019],
    # },
    # "etm_20": {
    #     "num_topics": 30,
    #     "epochs": 1_500,
    #     "train_embeddings": True,
    #     "isTestOnTrainingSet": True,
    #     "seeds": [2019, 2019, 2019, 2019, 2019],
    # },
    # "etm_21": {
    #     "num_topics": 40,
    #     "epochs": 300,
    #     "train_embeddings": False,
    #     "isTestOnTrainingSet": True,
    #     "seeds": [2019, 2019, 2019, 2019, 2019],
    # },
    # "etm_22": {
    #     "num_topics": 40,
    #     "epochs": 600,
    #     "train_embeddings": False,
    #     "isTestOnTrainingSet": True,
    #     "seeds": [2019, 2019, 2019, 2019, 2019],
    # },
    # "etm_23": {
    #     "num_topics": 40,
    #     "epochs": 900,
    #     "train_embeddings": False,
    #     "isTestOnTrainingSet": True,
    #     "seeds": [2019, 2019, 2019, 2019, 2019],
    # },
    # "etm_24": {
    #     "num_topics": 40,
    #     "epochs": 1_500,
    #     "train_embeddings": False,
    #     "isTestOnTrainingSet": True,
    #     "seeds": [2019, 2019, 2019, 2019, 2019],
    # },
    # "etm_25": {
    #     "num_topics": 40,
    #     "epochs": 1_500,
    #     "train_embeddings": False,
    #     "isTestOnTrainingSet": True,
    #     "seeds": [2019, 2019, 2019, 2019, 2019],
    # },
    # "etm_26": {
    #     "num_topics": 40,
    #     "epochs": 300,
    #     "train_embeddings": True,
    #     "isTestOnTrainingSet": True,
    #     "seeds": [2019, 2019, 2019, 2019, 2019],
    # },
    # "etm_27": {
    #     "num_topics": 40,
    #     "epochs": 600,
    #     "train_embeddings": True,
    #     "isTestOnTrainingSet": True,
    #     "seeds": [2019, 2019, 2019, 2019, 2019],
    # },
    # "etm_28": {
    #     "num_topics": 40,
    #     "epochs": 900,
    #     "train_embeddings": True,
    #     "isTestOnTrainingSet": True,
    #     "seeds": [2019, 2019, 2019, 2019, 2019],
    # },
    # "etm_29": {
    #     "num_topics": 40,
    #     "epochs": 1_500,
    #     "train_embeddings": True,
    #     "isTestOnTrainingSet": True,
    #     "seeds": [2019, 2019, 2019, 2019, 2019],
    # },
    # "etm_30": {
    #     "num_topics": 40,
    #     "epochs": 1_500,
    #     "train_embeddings": True,
    #     "isTestOnTrainingSet": True,
    #     "seeds": [2019, 2019, 2019, 2019, 2019],
    # },
    # Randomized trials for best results on training set (f-score, purity, npmi, uniques)
    # "etm_31": {
    #     "num_topics": 30,
    #     "epochs": 300,
    #     "train_embeddings": False,
    #     "isTestOnTrainingSet": True,
    #     "seeds": [2019, 1, 7, 28, 517],
    # },
    # "etm_32": {
    #     "num_topics": 30,
    #     "epochs": 300,
    #     "train_embeddings": True,
    #     "isTestOnTrainingSet": True,
    #     "seeds": [2019, 1, 7, 28, 517],
    # },
    # "etm_33": {
    #     "num_topics": 20,
    #     "epochs": 1_500,
    #     "train_embeddings": False,
    #     "isTestOnTrainingSet": True,
    #     "seeds": [2019, 1, 7, 28, 517],
    # },
    # "etm_34": {
    #     "num_topics": 20,
    #     "epochs": 1_500,
    #     "train_embeddings": True,
    #     "isTestOnTrainingSet": True,
    #     "seeds": [2019, 1, 7, 28, 517],
    # },
    # "etm_35": {
    #     "num_topics": 30,
    #     "epochs": 600,
    #     "train_embeddings": False,
    #     "isTestOnTrainingSet": True,
    #     "seeds": [2019, 1, 7, 28, 517],
    # },
    # "etm_36": {
    #     "num_topics": 30,
    #     "epochs": 600,
    #     "train_embeddings": True,
    #     "isTestOnTrainingSet": True,
    #     "seeds": [2019, 1, 7, 28, 517],
    # },
    # "etm_37": {
    #     "num_topics": 40,
    #     "epochs": 1_500,
    #     "train_embeddings": False,
    #     "isTestOnTrainingSet": True,
    #     "seeds": [2019, 1, 7, 28, 517],
    # },
    # "etm_38": {
    #     "num_topics": 40,
    #     "epochs": 1_500,
    #     "train_embeddings": True,
    #     "isTestOnTrainingSet": True,
    #     "seeds": [2019, 1, 7, 28, 517],
    # },
    # "etm_39": {
    #     "num_topics": 20,
    #     "epochs": 300,
    #     "train_embeddings": False,
    #     "isTestOnTrainingSet": True,
    #     "seeds": [2019, 1, 7, 28, 517],
    # },
    # "etm_40": {
    #     "num_topics": 20,
    #     "epochs": 300,
    #     "train_embeddings": True,
    #     "isTestOnTrainingSet": True,
    #     "seeds": [2019, 1, 7, 28, 517],
    # },
    # "etm_41": {
    #     "num_topics": 30,
    #     "epochs": 900,
    #     "train_embeddings": False,
    #     "isTestOnTrainingSet": True,
    #     "seeds": [2019, 1, 7, 28, 517],
    # },
    # "etm_42": {
    #     "num_topics": 30,
    #     "epochs": 900,
    #     "train_embeddings": True,
    #     "isTestOnTrainingSet": True,
    #     "seeds": [2019, 1, 7, 28, 517],
    # },
    # "etm_43": {
    #     "num_topics": 40,
    #     "epochs": 600,
    #     "train_embeddings": False,
    #     "isTestOnTrainingSet": True,
    #     "seeds": [2019, 1, 7, 28, 517],
    # },
    # "etm_44": {
    #     "num_topics": 40,
    #     "epochs": 600,
    #     "train_embeddings": True,
    #     "isTestOnTrainingSet": True,
    #     "seeds": [2019, 1, 7, 28, 517],
    # },
    # Randomized trials for best results on test set (f-score, purity, npmi, uniques)
    # "etm_45": {
    #     "num_topics": 30,
    #     "epochs": 300,
    #     "train_embeddings": False,
    #     "isTestOnTrainingSet": False,
    #     "seeds": [2019, 1, 7, 28, 517],
    # },
    # "etm_46": {
    #     "num_topics": 30,
    #     "epochs": 300,
    #     "train_embeddings": True,
    #     "isTestOnTrainingSet": False,
    #     "seeds": [2019, 1, 7, 28, 517],
    # },
    # "etm_47": {
    #     "num_topics": 20,
    #     "epochs": 1_500,
    #     "train_embeddings": False,
    #     "isTestOnTrainingSet": False,
    #     "seeds": [2019, 1, 7, 28, 517],
    # },
    # "etm_48": {
    #     "num_topics": 20,
    #     "epochs": 1_500,
    #     "train_embeddings": True,
    #     "isTestOnTrainingSet": False,
    #     "seeds": [2019, 1, 7, 28, 517],
    # },
    # "etm_49": {
    #     "num_topics": 30,
    #     "epochs": 600,
    #     "train_embeddings": False,
    #     "isTestOnTrainingSet": False,
    #     "seeds": [2019, 1, 7, 28, 517],
    # },
    # "etm_50": {
    #     "num_topics": 30,
    #     "epochs": 600,
    #     "train_embeddings": True,
    #     "isTestOnTrainingSet": False,
    #     "seeds": [2019, 1, 7, 28, 517],
    # },
    # "etm_51": {
    #     "num_topics": 40,
    #     "epochs": 1_500,
    #     "train_embeddings": False,
    #     "isTestOnTrainingSet": False,
    #     "seeds": [2019, 1, 7, 28, 517],
    # },
    # "etm_52": {
    #     "num_topics": 40,
    #     "epochs": 1_500,
    #     "train_embeddings": True,
    #     "isTestOnTrainingSet": False,
    #     "seeds": [2019, 1, 7, 28, 517],
    # },
    # "etm_53": {
    #     "num_topics": 20,
    #     "epochs": 300,
    #     "train_embeddings": False,
    #     "isTestOnTrainingSet": False,
    #     "seeds": [2019, 1, 7, 28, 517],
    # },
    # "etm_54": {
    #     "num_topics": 20,
    #     "epochs": 300,
    #     "train_embeddings": True,
    #     "isTestOnTrainingSet": False,
    #     "seeds": [2019, 1, 7, 28, 517],
    # },
    # "etm_55": {
    #     "num_topics": 30,
    #     "epochs": 900,
    #     "train_embeddings": False,
    #     "isTestOnTrainingSet": False,
    #     "seeds": [2019, 1, 7, 28, 517],
    # },
    # "etm_56": {
    #     "num_topics": 30,
    #     "epochs": 900,
    #     "train_embeddings": True,
    #     "isTestOnTrainingSet": False,
    #     "seeds": [2019, 1, 7, 28, 517],
    # },
    # "etm_57": {
    #     "num_topics": 40,
    #     "epochs": 600,
    #     "train_embeddings": False,
    #     "isTestOnTrainingSet": False,
    #     "seeds": [2019, 1, 7, 28, 517],
    # },
    # "etm_58": {
    #     "num_topics": 40,
    #     "epochs": 600,
    #     "train_embeddings": True,
    #     "isTestOnTrainingSet": False,
    #     "seeds": [2019, 1, 7, 28, 517],
    # },
    # Randomized trials for best results on full set (f-score, purity, npmi, uniques)
    # "etm_59": {
    #     "num_topics": 20,
    #     "epochs": 300,
    #     "train_embeddings": False,
    #     "isTestOnTrainingSet": True,
    #     "seeds": [2019, 1, 7, 28, 517],
    # },
    # "etm_60": {
    #     "num_topics": 30,
    #     "epochs": 900,
    #     "train_embeddings": False,
    #     "isTestOnTrainingSet": True,
    #     "seeds": [2019, 1, 7, 28, 517],
    # },
    # "etm_61": {
    #     "num_topics": 40,
    #     "epochs": 1_500,
    #     "train_embeddings": False,
    #     "isTestOnTrainingSet": True,
    #     "seeds": [2019, 1, 7, 28, 517],
    # },
    # Configuration of models to measure training time
    "etm_62": {
        "num_topics": 40,
        "epochs": 900,
        "train_embeddings": False,
        "isTestOnTrainingSet": True,
        "seeds": [2019, 1, 7, 28, 517],
    },
    "etm_63": {
        "num_topics": 40,
        "epochs": 600,
        "train_embeddings": False,
        "isTestOnTrainingSet": True,
        "seeds": [2019, 1, 7, 28, 517],
    },
    "etm_64": {
        "num_topics": 40,
        "epochs": 300,
        "train_embeddings": False,
        "isTestOnTrainingSet": True,
        "seeds": [2019, 1, 7, 28, 517],
    },
}

# Tested configuration for ag dataset (5000 features) - systematic grid search
configurations_ag: dict = {
    # No randomized trials i.e. fixed seeds to take best results, i.e. models on training set (
    # f-score, purity, npmi, uniques) to next deeper fine-tuning (parametrization).
    # For configurations from etm_1 to etm_24 set STEPS_RANGE_NUMBER = 1
    # to faster perform computation.
    # It is not influence on final results, i.e. same seed produce same results in ech time
    # "etm_1": {
    #     "num_topics": 20,
    #     "epochs": 300,
    #     "train_embeddings": False,
    #     "isTestOnTrainingSet": True,
    #     "seeds": [2019, 2019, 2019, 2019, 2019],
    # },
    # "etm_2": {
    #     "num_topics": 20,
    #     "epochs": 600,
    #     "train_embeddings": False,
    #     "isTestOnTrainingSet": True,
    #     "seeds": [2019, 2019, 2019, 2019, 2019],
    # },
    # "etm_3": {
    #     "num_topics": 20,
    #     "epochs": 900,
    #     "train_embeddings": False,
    #     "isTestOnTrainingSet": True,
    #     "seeds": [2019, 2019, 2019, 2019, 2019],
    # },
    # "etm_4": {
    #     "num_topics": 20,
    #     "epochs": 1_500,
    #     "train_embeddings": False,
    #     "isTestOnTrainingSet": True,
    #     "seeds": [2019, 2019, 2019, 2019, 2019],
    # },
    # "etm_5": {
    #     "num_topics": 20,
    #     "epochs": 300,
    #     "train_embeddings": True,
    #     "isTestOnTrainingSet": True,
    #     "seeds": [2019, 2019, 2019, 2019, 2019],
    # },
    # "etm_6": {
    #     "num_topics": 20,
    #     "epochs": 600,
    #     "train_embeddings": True,
    #     "isTestOnTrainingSet": True,
    #     "seeds": [2019, 2019, 2019, 2019, 2019],
    # },
    # "etm_7": {
    #     "num_topics": 20,
    #     "epochs": 900,
    #     "train_embeddings": True,
    #     "isTestOnTrainingSet": True,
    #     "seeds": [2019, 2019, 2019, 2019, 2019],
    # },
    # "etm_8": {
    #     "num_topics": 20,
    #     "epochs": 1_500,
    #     "train_embeddings": True,
    #     "isTestOnTrainingSet": True,
    #     "seeds": [2019, 2019, 2019, 2019, 2019],
    # },
    # "etm_9": {
    #     "num_topics": 30,
    #     "epochs": 300,
    #     "train_embeddings": False,
    #     "isTestOnTrainingSet": True,
    #     "seeds": [2019, 2019, 2019, 2019, 2019],
    # },
    # "etm_10": {
    #     "num_topics": 30,
    #     "epochs": 600,
    #     "train_embeddings": False,
    #     "isTestOnTrainingSet": True,
    #     "seeds": [2019, 2019, 2019, 2019, 2019],
    # },
    # "etm_11": {
    #     "num_topics": 30,
    #     "epochs": 900,
    #     "train_embeddings": False,
    #     "isTestOnTrainingSet": True,
    #     "seeds": [2019, 2019, 2019, 2019, 2019],
    # },
    # "etm_12": {
    #     "num_topics": 30,
    #     "epochs": 1_500,
    #     "train_embeddings": False,
    #     "isTestOnTrainingSet": True,
    #     "seeds": [2019, 2019, 2019, 2019, 2019],
    # },
    # "etm_13": {
    #     "num_topics": 30,
    #     "epochs": 300,
    #     "train_embeddings": True,
    #     "isTestOnTrainingSet": True,
    #     "seeds": [2019, 2019, 2019, 2019, 2019],
    # },
    # "etm_14": {
    #     "num_topics": 30,
    #     "epochs": 600,
    #     "train_embeddings": True,
    #     "isTestOnTrainingSet": True,
    #     "seeds": [2019, 2019, 2019, 2019, 2019],
    # },
    # "etm_15": {
    #     "num_topics": 30,
    #     "epochs": 900,
    #     "train_embeddings": True,
    #     "isTestOnTrainingSet": True,
    #     "seeds": [2019, 2019, 2019, 2019, 2019],
    # },
    # "etm_16": {
    #     "num_topics": 30,
    #     "epochs": 1_500,
    #     "train_embeddings": True,
    #     "isTestOnTrainingSet": True,
    #     "seeds": [2019, 2019, 2019, 2019, 2019],
    # },
    # "etm_17": {
    #     "num_topics": 40,
    #     "epochs": 300,
    #     "train_embeddings": False,
    #     "isTestOnTrainingSet": True,
    #     "seeds": [2019, 2019, 2019, 2019, 2019],
    # },
    # "etm_18": {
    #     "num_topics": 40,
    #     "epochs": 600,
    #     "train_embeddings": False,
    #     "isTestOnTrainingSet": True,
    #     "seeds": [2019, 2019, 2019, 2019, 2019],
    # },
    # "etm_19": {
    #     "num_topics": 40,
    #     "epochs": 900,
    #     "train_embeddings": False,
    #     "isTestOnTrainingSet": True,
    #     "seeds": [2019, 2019, 2019, 2019, 2019],
    # },
    # "etm_20": {
    #     "num_topics": 40,
    #     "epochs": 1_500,
    #     "train_embeddings": False,
    #     "isTestOnTrainingSet": True,
    #     "seeds": [2019, 2019, 2019, 2019, 2019],
    # },
    # "etm_21": {
    #     "num_topics": 40,
    #     "epochs": 300,
    #     "train_embeddings": True,
    #     "isTestOnTrainingSet": True,
    #     "seeds": [2019, 2019, 2019, 2019, 2019],
    # },
    # "etm_22": {
    #     "num_topics": 40,
    #     "epochs": 600,
    #     "train_embeddings": True,
    #     "isTestOnTrainingSet": True,
    #     "seeds": [2019, 2019, 2019, 2019, 2019],
    # },
    # "etm_23": {
    #     "num_topics": 40,
    #     "epochs": 900,
    #     "train_embeddings": True,
    #     "isTestOnTrainingSet": True,
    #     "seeds": [2019, 2019, 2019, 2019, 2019],
    # },
    # "etm_24": {
    #     "num_topics": 40,
    #     "epochs": 1_500,
    #     "train_embeddings": True,
    #     "isTestOnTrainingSet": True,
    #     "seeds": [2019, 2019, 2019, 2019, 2019],
    # },
    # Randomized trials for best results on training set (f-score, purity, npmi, uniques)
    # "etm_25": {
    #     "num_topics": 40,
    #     "epochs": 900,
    #     "train_embeddings": True,
    #     "isTestOnTrainingSet": True,
    #     "seeds": [2019, 1, 7, 28, 517],
    # },
    # "etm_26": {
    #     "num_topics": 20,
    #     "epochs": 600,
    #     "train_embeddings": False,
    #     "isTestOnTrainingSet": True,
    #     "seeds": [2019, 1, 7, 28, 517],
    # },
    # "etm_27": {
    #     "num_topics": 30,
    #     "epochs": 300,
    #     "train_embeddings": False,
    #     "isTestOnTrainingSet": True,
    #     "seeds": [2019, 1, 7, 28, 517],
    # },
    # "etm_28": {
    #     "num_topics": 40,
    #     "epochs": 600,
    #     "train_embeddings": False,
    #     "isTestOnTrainingSet": True,
    #     "seeds": [2019, 1, 7, 28, 517],
    # },
    # "etm_29": {
    #     "num_topics": 20,
    #     "epochs": 900,
    #     "train_embeddings": False,
    #     "isTestOnTrainingSet": True,
    #     "seeds": [2019, 1, 7, 28, 517],
    # },
    # "etm_30": {
    #     "num_topics": 30,
    #     "epochs": 1_500,
    #     "train_embeddings": False,
    #     "isTestOnTrainingSet": True,
    #     "seeds": [2019, 1, 7, 28, 517],
    # },
    # "etm_31": {
    #     "num_topics": 40,
    #     "epochs": 900,
    #     "train_embeddings": False,
    #     "isTestOnTrainingSet": True,
    #     "seeds": [2019, 1, 7, 28, 517],
    # },
    # # Randomized trials for best results on test set (f-score, purity, npmi, uniques)
    # "etm_32": {
    #     "num_topics": 40,
    #     "epochs": 900,
    #     "train_embeddings": True,
    #     "isTestOnTrainingSet": False,
    #     "seeds": [2019, 1, 7, 28, 517],
    # },
    # "etm_33": {
    #     "num_topics": 20,
    #     "epochs": 600,
    #     "train_embeddings": False,
    #     "isTestOnTrainingSet": False,
    #     "seeds": [2019, 1, 7, 28, 517],
    # },
    # "etm_34": {
    #     "num_topics": 30,
    #     "epochs": 300,
    #     "train_embeddings": False,
    #     "isTestOnTrainingSet": False,
    #     "seeds": [2019, 1, 7, 28, 517],
    # },
    # "etm_35": {
    #     "num_topics": 40,
    #     "epochs": 600,
    #     "train_embeddings": False,
    #     "isTestOnTrainingSet": False,
    #     "seeds": [2019, 1, 7, 28, 517],
    # },
    # "etm_36": {
    #     "num_topics": 20,
    #     "epochs": 900,
    #     "train_embeddings": False,
    #     "isTestOnTrainingSet": False,
    #     "seeds": [2019, 1, 7, 28, 517],
    # },
    # "etm_37": {
    #     "num_topics": 30,
    #     "epochs": 1_500,
    #     "train_embeddings": False,
    #     "isTestOnTrainingSet": False,
    #     "seeds": [2019, 1, 7, 28, 517],
    # },
    # "etm_38": {
    #     "num_topics": 40,
    #     "epochs": 900,
    #     "train_embeddings": False,
    #     "isTestOnTrainingSet": False,
    #     "seeds": [2019, 1, 7, 28, 517],
    # },
    # Randomized trials for best results on full set (f-score, purity, npmi, uniques)
    # "etm_39": {
    #     "num_topics": 20,
    #     "epochs": 900,
    #     "train_embeddings": False,
    #     "isTestOnTrainingSet": True,
    #     "seeds": [2019, 1, 7, 28, 517],
    # },
    # "etm_40": {
    #     "num_topics": 30,
    #     "epochs": 300,
    #     "train_embeddings": False,
    #     "isTestOnTrainingSet": True,
    #     "seeds": [2019, 1, 7, 28, 517],
    # },
    "etm_41": {
        "num_topics": 40,
        "epochs": 600,
        "train_embeddings": False,
        "isTestOnTrainingSet": True,
        "seeds": [2019, 1, 7, 28, 517],
    },
    # Configuration of models to measure training time
    "etm_43": {
        "num_topics": 40,
        "epochs": 900,
        "train_embeddings": False,
        "isTestOnTrainingSet": True,
        "seeds": [2019, 1, 7, 28, 517],
    },
    "etm_44": {
        "num_topics": 40,
        "epochs": 300,
        "train_embeddings": False,
        "isTestOnTrainingSet": True,
        "seeds": [2019, 1, 7, 28, 517],
    },
}

# Create a dictionary to store all configurations
all_configurations = {
    "20news": configurations_20news,
    "bbc": configurations_bbc,
    "ag": configurations_ag,
}

# Choose the dataset based on the DATA_SET variable
if DATA_SET in all_configurations:
    # Loading tested configuration for given dataset
    configurations = all_configurations[DATA_SET]
else:
    raise ValueError(f"Invalid DATA_SET: {DATA_SET}")


def main() -> int:
    """
    Build a topic model using ETM method and make statistic
    of the clustering based on the created topic representation.

    Returns:
        int: 0
    """
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)

    data_set: DatasetInterface = DatasetLoader(
        DATA_SET, FEATURE_LIMIT, DATA_SET_PATH
    ).load_dataset()

    train_data = [" ".join(doc) for doc in data_set.train_tokens()]
    test_data = [" ".join(doc) for doc in data_set.test_tokens()]

    if len(train_data) == len(test_data):
        print(
            f"Case of full data training: "
            f"len(train_data) {len(train_data)} == len(train_data) {len(test_data)}"
        )
        all_data = train_data
    else:
        print(
            f"Case of not full data training: "
            f"len(train_data) {len(train_data)} != len(train_data) {len(test_data)}"
        )
        all_data = train_data + test_data

    vocabulary, train_dataset, test_dataset = create_etm_datasets(
        all_data, min_df=0, max_df=100_000, train_size=len(all_data), debug_mode=True
    )

    if train_dataset.get("tokens").shape[0] != len(train_data):  # type: ignore
        raise ValueError("Numbers are different for training set.")
    if test_dataset.get("test").get("tokens").shape[0] != len(test_data):  # type: ignore
        if len(train_data) != len(test_data):
            raise ValueError("Numbers are different for test set.")

    # Training word2vec embeddings
    embeddings_mapping = embedding.create_word2vec_embedding_from_dataset(train_data)

    results: dict = {}
    configurations_training_times = []

    for configuration_number in list(configurations.keys()):
        print(f"Configuration number: {configuration_number}")
        configuration: Dict[str, Any] = configurations.get(configuration_number, {})

        N = configuration.get("num_topics")
        training_times = []

        for iteration_number in range(STEPS_RANGE_NUMBER):
            model_name = f"{configuration_number}_{N}_{iteration_number}"
            print(f"Learning model: {model_name}")
            seed: int = configuration.get("seeds", [])[iteration_number]

            # Training an ETM instance
            etm_instance = ETM(
                vocabulary,
                embeddings=embeddings_mapping,  # You can pass here the path to a word2vec file
                # or a KeyedVectors instance
                num_topics=configuration.get("num_topics"),
                epochs=configuration.get("epochs"),
                debug_mode=True,
                train_embeddings=configuration.get("train_embeddings"),
                # Optional. If True, ETM will learn word embeddings jointly with topic
                # embeddings. By default, is False. If 'embeddings' argument is being passed,
                # this argument must not be True
                seed=seed,
            )
            start_time = timeit.default_timer()
            etm_instance.fit(train_dataset)
            end_time = timeit.default_timer()
            time_difference = end_time - start_time
            training_times.append(time_difference)
            print(f"Training time (seconds): {time_difference}")

            topics = etm_instance.get_topics(N)
            topic_coherence = etm_instance.get_topic_coherence()
            topic_diversity = etm_instance.get_topic_diversity()

            conf_details = {
                "topics": topics,
                "topic_coherence": topic_coherence,
                "topic_diversity": topic_diversity,
            }
            results[configuration_number] = conf_details

            with open(os.path.join(OUTPUT_PATH, model_name), "wb") as file:
                pkl.dump(etm_instance, file)

            train_norm = etm_instance.get_document_topic_dist().cpu().numpy()

            if configuration.get("isTestOnTrainingSet"):
                print("ETM is testing on the training set")
                test_norm = train_norm
            else:
                print("ETM is testing on the test set")
                test_norm = (
                    get_test_document_topic_dist(etm_instance, test_dataset)
                    .cpu()
                    .numpy()
                )

            clustering_met: RetrivalMetrics = RetrivalMetrics(
                model_name,
                N,
                train_norm,
                test_norm,
                data_set.train_labels(),
                data_set.train_labels()
                if configuration.get("isTestOnTrainingSet")
                else data_set.test_labels(),
                data_set.categories(),
            )
            clustering_met.calculate_metrics()

            print(f"Purity ETM: {clustering_met.purity}")
            print(f"F-score ETM: {clustering_met.classification_metrics.fscore}")

            clustering_met.save(OUTPUT_PATH, f"{model_name}")

        configurations_training_times.append(training_times)

    etm_results_model_file = os.path.join(OUTPUT_PATH, "etm_results.json")

    # Save the dictionary to a JSON file
    with open(etm_results_model_file, "w", encoding="utf-8") as json_file:
        json.dump(results, json_file)

    del results

    # Load the dictionary from the JSON file
    with open(etm_results_model_file, "r", encoding="utf-8") as json_file:
        results = json.load(json_file)

    # Extract the configurations values from the dictionary
    topic_configurations_list = sorted(list(configurations.keys()))

    # Extract the topic coherence values from the nested dictionary
    topic_coherence_list = [result["topic_coherence"] for result in results.values()]

    # Extract the topic diversity values from the nested dictionary
    topic_diversity_list = [result["topic_diversity"] for result in results.values()]

    # Create a DataFrame with the data
    data = pd.DataFrame(
        {
            "Configuration": topic_configurations_list,
            "Topic Coherence": topic_coherence_list,
            "Topic Diversity": topic_diversity_list,
        }
    )
    data["Configuration"] = pd.Categorical(
        data["Configuration"], categories=data["Configuration"].tolist(), ordered=True
    )

    # Create the ggplot line plot
    # %matplotlib inline
    scatter_plot = (
        ggplot(data, aes(x="Configuration", y="Topic Coherence"))
        + geom_point()
        + labs(x="Configuration", y="Topic Coherence")
        + theme(axis_text_x=element_text(angle=90, hjust=1))
    )

    # Save the plot
    scatter_plot.save(os.path.join(OUTPUT_PATH, "etm_scatter_plot_tc.png"))

    # Create the ggplot line plot
    scatter_plot = (
        ggplot(data, aes(x="Configuration", y="Topic Diversity"))
        + geom_point()
        + labs(x="Configuration", y="Topic Diversity")
        + theme(axis_text_x=element_text(angle=90, hjust=1))
    )

    # Save the plot
    scatter_plot.save(os.path.join(OUTPUT_PATH, "etm_scatter_plot_td.png"))

    return 0


if __name__ == "__main__":
    print("ETM_model_script.py is being run directly")
else:
    print("ETM_model_script.py is being imported into another module")
