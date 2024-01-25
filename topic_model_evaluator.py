"""
Topic Model Evaluation Module

This module handles the evaluation of topic models using various metrics. It includes functions
to load different types of topic models, calculate topic-related metrics, and evaluate the models
using coherence scores.
"""
import csv
import os

import numpy as np
import pandas as pd
import pickle as pkl
from gensim.models import LdaModel

from model.configuration.etm_configuration_context import ETMConfigurationContext
from model.network.stm_model_runner import STMModelRunner
from model.topic.topic_metric_factory import TopicMetricsFactory
from model.topic.topic_metrics import TopicMetrics

from topic_model_builder_full_configurations_ETM import (
    configurations as EtmModelsConfigurations,
)
from plotnine import ggplot, aes, geom_point, labs, theme, element_text


def load_stm_model(_model_name):
    """
    Load pre-trained STM model
    """
    return STMModelRunner.load(OUTPUT_PATH, _model_name)


def load_lda_model(_model_name):
    """
    Load pre-trained LDA model
    """
    lda_model_file = os.path.join(OUTPUT_PATH, _model_name)
    return LdaModel.load(lda_model_file)


def load_btm_model(_model_name):
    """
    Load pre-trained BTM model
    """
    btm_model_file = os.path.join(OUTPUT_PATH, _model_name)
    with open(btm_model_file, "rb") as file:
        return pkl.load(file)


def load_etm_model(_model_name):
    """
    Load pre-trained ETM model
    """
    etm_model_file = os.path.join(OUTPUT_PATH, _model_name)
    with open(etm_model_file, "rb") as file:
        return pkl.load(file)


def get_model(model_type, _model_name):
    """
    Load and return a specific type of model.

    Args:
        model_type (str): The type of the model to load.
            Possible values: "STM", "lda_auto", "btm", "etm".
        _model_name (str): The name or identifier of the model to load.

    Returns:
        model: The loaded model of the specified type and name.

    Raises:
        ValueError: If the specified model_type is not recognized.
    """
    model_loaders = {
        "STM": load_stm_model,
        "lda_auto": load_lda_model,
        "btm": load_btm_model,
        "etm": load_etm_model,
    }

    if "_" in model_type:
        model_type = model_type.split("_")[0]

    loader = model_loaders.get(model_type)
    if loader:
        return loader(_model_name)

    raise ValueError("Unknown model type: " + model_type)


def calculate_unique(topics):
    """
    Calculate the uniqueness score of a list of topics.

    This function calculates the uniqueness score of a list of topics based on the uniqueness of
    words used across all topics.

    Args:
        topics (list): A list of topic objects, where each topic object has a 'words' attribute
            representing the words associated with that topic.

    Returns:
        float: The uniqueness score, calculated as the ratio of unique words used in all topics
            to the total number of words in all topics. The score is normalized by dividing by ten.

    Example:
        topic1 = {'words': ['apple', 'banana', 'orange']}
        topic2 = {'words': ['banana', 'grape', 'kiwi']}
        topics = [topic1, topic2]
        score = calculate_unique(topics)
        # score will be (4 / (6 * 10)) = 0.066666...
    """
    uniq = set()
    for topic in topics:
        uniq.update(topic.words)
    return len(uniq) / (len(topics) * 10)


# Currently we have default datasets such as: 20news, ag, and bbc
DATA_SET = "ag"

INPUT_PATH = f"model-input-data/{DATA_SET}"
CONFIG_PATH = "network-configuration"
OUTPUT_PATH = f"model-output-data/{DATA_SET}"

is_extend: bool = True  # Set to true if you want to compute metrics for ETM model
MODELS_RANG_NAMES = list(EtmModelsConfigurations.keys()) if is_extend else []
MODELS_RANG_NAMES = (
    MODELS_RANG_NAMES + []
)  # Fill this list using appropriately models names, i.e. STM, lda_auto, or btm
TOPICS_RANGE_NUMBER = [20, 30, 40]
STEPS_RANGE_NUMBER = 5

REMOTE_NDPOINT_PALMETTO = True

if REMOTE_NDPOINT_PALMETTO:
    ENDPOINT_PALMETTO = "http://palmetto.aksw.org/palmetto-webapp/service/"
else:
    ENDPOINT_PALMETTO = "http://localhost:7777/service/"


def main() -> int:
    with open(
        os.path.join(OUTPUT_PATH, "_coherence-results.csv"), "w+", encoding="utf-8"
    ) as coherence_results_csv:
        writer = csv.writer(coherence_results_csv, dialect="unix")
        header = ["DS", "N", "model", "TopN", "metric", "metric_val"]
        writer.writerow(header)

        for i in range(STEPS_RANGE_NUMBER):
            for m in MODELS_RANG_NAMES:
                with ETMConfigurationContext(m, TOPICS_RANGE_NUMBER):
                    print(f"Computation for range: {m} {TOPICS_RANGE_NUMBER}")
                    for N in TOPICS_RANGE_NUMBER:
                        model_name = f"{m}_{N}_{i}"
                        print(f"Computation for model: {m}: {model_name}")

                        model = get_model(m, model_name)
                        metrics_file_name = f"{model_name}_topic_metrics"

                        metrics: TopicMetrics

                        if os.path.exists(os.path.join(OUTPUT_PATH, metrics_file_name)):
                            metrics = TopicMetrics.load(
                                OUTPUT_PATH, f"{model_name}_topic_metrics"
                            )
                        else:
                            metrics = TopicMetricsFactory.get_metric(
                                m, N, model, ENDPOINT_PALMETTO
                            )
                            metrics.generate_metrics()
                            metrics.save(OUTPUT_PATH, f"{model_name}_topic_metrics")

                        metrics = TopicMetrics.load(
                            OUTPUT_PATH, f"{model_name}_topic_metrics"
                        )

                        top = sorted(
                            metrics.topics,
                            key=lambda x: x.get_metric("npmi"),
                            reverse=True,
                        )
                        for t in metrics.topics:
                            print(
                                f"Metric: {t.get_metric('npmi')} {t.get_metric('ca')} for:"
                                f" {t.words}"
                            )
                        for top_n in [100, 75]:
                            for metric in metrics.coherence_metrics:
                                result_line = [
                                    DATA_SET,
                                    N,
                                    m,
                                    top_n,
                                    metric,
                                    round(
                                        metrics.get_average_metric_for_top_n(
                                            metric, top_n
                                        ),
                                        3,
                                    ),
                                ]
                                print(f"Results: {top_n} {metric} {result_line}")
                                writer.writerow(result_line)
                                coherence_results_csv.flush()

    # Show coherence results and put them into data frame
    df_coherence = pd.read_csv(os.path.join(OUTPUT_PATH, "_coherence-results.csv"))

    npmi: dict = {m: [] for m in MODELS_RANG_NAMES}

    for model in MODELS_RANG_NAMES:
        mean_all_N = []

        with ETMConfigurationContext(model, TOPICS_RANGE_NUMBER):
            print(f"Computation for range: {model} {TOPICS_RANGE_NUMBER}")

            for N in TOPICS_RANGE_NUMBER:
                for metric in ["npmi"]:
                    metric_record = df_coherence[
                        (df_coherence["model"] == model)
                        & (df_coherence["N"] == N)
                        & (df_coherence["metric"] == metric)
                        & (df_coherence["TopN"] == 100)
                        & (df_coherence["DS"] == DATA_SET)
                    ]["metric_val"]
                    print(model, N, metric, round(metric_record.mean(), 3))
                    mean_all_N.append(round(metric_record.mean(), 3))
                    npmi[model].append(mean_all_N)
            print("avg ", model, N, metric, round(np.average(mean_all_N), 3))

    # Create the ggplot line plot
    # %matplotlib inline
    df_coherence_npmi = pd.DataFrame()

    for key in npmi.keys():
        new_row = {
            "Configuration": key,
            "NPMI average": npmi.get(key, [])[0][0],
        }
        df_coherence_npmi = df_coherence_npmi.append(new_row, ignore_index=True)

    df_coherence_npmi["Configuration"] = pd.Categorical(
        df_coherence_npmi["Configuration"],
        categories=df_coherence_npmi["Configuration"].tolist(),
        ordered=True,
    )

    scatter_plot = (
        ggplot(df_coherence_npmi, aes(x="Configuration", y="NPMI average"))
        + geom_point()
        + labs(x="Configuration", y="NPMI average")
        + theme(axis_text_x=element_text(angle=90, hjust=1))
    )
    scatter_plot.save(os.path.join(OUTPUT_PATH, "etm_scatter_plot_npmi-avg.png"))

    # There are no directly saved uniques, so we extract it from serialized objects
    uniques: dict = {m: [] for m in MODELS_RANG_NAMES}

    for i in range(STEPS_RANGE_NUMBER):
        for m in MODELS_RANG_NAMES:
            with ETMConfigurationContext(m, TOPICS_RANGE_NUMBER):
                print(f"Computation for range: {m} {TOPICS_RANGE_NUMBER}")
                for N in TOPICS_RANGE_NUMBER:
                    model_name = f"{m}_{N}_{i}"
                    print(f"Computation for model: {m}: {model_name}")
                    model = get_model(m, model_name)
                    metrics_file_name = f"{model_name}_topic_metrics"

                    metrics: TopicMetrics  # type: ignore

                    file_path = os.path.join(OUTPUT_PATH, metrics_file_name)

                    if os.path.exists(file_path):
                        metrics = TopicMetrics.load(
                            OUTPUT_PATH, f"{model_name}_topic_metrics"
                        )
                    else:
                        raise FileNotFoundError(
                            f"The file '{file_path}' does not exist."
                        )

                    uniques[m].append(calculate_unique(metrics.topics))

    df_uniques = pd.DataFrame()

    for m in MODELS_RANG_NAMES:
        with ETMConfigurationContext(m, TOPICS_RANGE_NUMBER):
            for N in TOPICS_RANGE_NUMBER:
                average_uniques = np.average(uniques[m]) * 100
                std_dev_uniques = np.std(uniques[m]) * 100
                print(
                    f"Uniques for {m} and {N} ({TOPICS_RANGE_NUMBER}):"
                    f"{average_uniques:.2f} ({std_dev_uniques:.2f}))"
                )

                new_row = {
                    "Configuration": m,
                    "No. topics": N,
                    "Uniques average": average_uniques,
                    "Uniques standard deviation": std_dev_uniques,
                }

                df_uniques = df_uniques.append(new_row, ignore_index=True)

    df_uniques[["No. topics"]] = df_uniques[["No. topics"]].astype(int)

    df_uniques["Configuration"] = pd.Categorical(
        df_uniques["Configuration"],
        categories=df_uniques["Configuration"].tolist(),
        ordered=True,
    )

    # Create the ggplot line plot
    # %matplotlib inline
    scatter_plot = (
        ggplot(df_uniques, aes(x="Configuration", y="Uniques average"))
        + geom_point()
        + labs(x="Configuration", y="Uniques average")
        + theme(axis_text_x=element_text(angle=90, hjust=1))
    )
    scatter_plot.save(os.path.join(OUTPUT_PATH, "etm_scatter_plot_uniques-avg.png"))

    # Uniques-NMPI plot
    df_uniques_npmi = df_uniques.merge(
        df_coherence_npmi, left_on="Configuration", right_on="Configuration"
    )

    scatter_plot = (
        ggplot(df_uniques_npmi, aes(x="Uniques average", y="NPMI average"))
        + geom_point()
        + labs(x="Uniques average", y="NPMI average")
        + theme(axis_text_x=element_text(angle=90, hjust=1))
    )
    scatter_plot.save(
        os.path.join(OUTPUT_PATH, "etm_scatter_plot_uniques-npmi-avg.png")
    )

    return 0


if __name__ == "__main__":
    print("topic_model_evaluator.py is being run directly")
else:
    print("topic_model_evaluator.py is being imported into another module")
