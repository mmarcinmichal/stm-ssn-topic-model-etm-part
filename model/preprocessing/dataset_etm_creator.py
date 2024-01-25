import random
from typing import Tuple, List

import nltk
import numpy as np
from embedded_topic_model.utils.preprocessing import (
    _create_dictionaries,
    _remove_empty_documents,
    _create_list_words,
    _create_document_indices,
    _create_bow,
    _split_bow,
    _to_numpy_array,
)
from sklearn.feature_extraction.text import CountVectorizer

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
