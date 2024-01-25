import os

import numpy as np
import pandas as pd

from model.configuration.etm_configuration_context import ETMConfigurationContext
from model.evaluation.retrival_metrics import RetrivalMetrics
from model.topic.topic_metrics import TopicMetrics
from topic_model_builder_full_configurations_ETM import (
    configurations as EtmModelsConfigurations,
)
from topic_model_evaluator import get_model, calculate_unique

# Currently we have default datasets such as: 20news, ag, and bbc
DATA_SET = "20news"

INPUT_PATH = f"model-input-data/{DATA_SET}"
CONFIG_PATH = "network-configuration"
OUTPUT_PATH = f"model-output-data/{DATA_SET}"

is_extend: bool = True  # Set to true if you want to compute metrics for ETM model
MODELS_RANG_NAMES = list(EtmModelsConfigurations.keys()) if is_extend else []
TOPICS_RANGE_NUMBER = [20, 30, 40]
STEPS_RANGE_NUMBER = 5


def main() -> int:
    # Load data about Purity
    purity: dict = {m: [] for m in MODELS_RANG_NAMES}

    for i in range(STEPS_RANGE_NUMBER):
        for m in MODELS_RANG_NAMES:
            with ETMConfigurationContext(m, TOPICS_RANGE_NUMBER):
                print(f"Computation for range: {m} {TOPICS_RANGE_NUMBER}")
                for N in TOPICS_RANGE_NUMBER:
                    model_name = f"{m}_{N}_{i}_clustering_metrics"
                    print(f"Computation model_name: {model_name}")
                    M: RetrivalMetrics = RetrivalMetrics.load(OUTPUT_PATH, model_name)
                    print(model_name, M.purity)
                    purity[m].append(M.purity)

    df_retrieval_metrics = pd.DataFrame()

    for m in MODELS_RANG_NAMES:
        with ETMConfigurationContext(m, TOPICS_RANGE_NUMBER):
            for N in TOPICS_RANGE_NUMBER:
                average_purity_score = np.average(purity[m]) * 100
                std_dev_purity_score = np.std(purity[m]) * 100
                print(
                    f"Purity for {m} and {N} ({TOPICS_RANGE_NUMBER}):"
                    f"{average_purity_score:.2f} ({std_dev_purity_score:.2f}))"
                )

                new_row = {
                    "Configuration": m,
                    "No. topics": N,
                    "Purity average": average_purity_score,
                    "Purity standard deviation": std_dev_purity_score,
                }

                df_retrieval_metrics = df_retrieval_metrics.append(
                    new_row, ignore_index=True
                )

    df_retrieval_metrics[["No. topics"]] = df_retrieval_metrics[["No. topics"]].astype(
        int
    )

    df_retrieval_metrics["Configuration"] = pd.Categorical(
        df_retrieval_metrics["Configuration"],
        categories=df_retrieval_metrics["Configuration"].tolist(),
        ordered=True,
    )

    # Load data about NPMI or C_a
    df_coherence = pd.read_csv(os.path.join(OUTPUT_PATH, "_coherence-results.csv"))

    # Group by columns 'DS', 'N', 'model', and 'TopN', and calculate the average of 'metric_val'
    grouped_df = (
        df_coherence.groupby(["DS", "N", "model", "TopN", "metric"])["metric_val"]
        .mean()
        .reset_index()
    )
    pivot_df = grouped_df.pivot_table(
        index=["DS", "N", "model", "TopN"], columns="metric", values="metric_val"
    ).reset_index()

    # TODO :: Replace code bellow to compute npmi and ca with the simple code above
    npmi: dict = {m: [] for m in MODELS_RANG_NAMES}
    ca: dict = {m: [] for m in MODELS_RANG_NAMES}

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

    df_coherence_npmi = pd.DataFrame()

    for key in npmi.keys():
        new_row = {
            "Configuration": key,
            "No. topics": EtmModelsConfigurations.get(key, []).get("num_topics"),
            "NPMI average": npmi.get(key, [])[0][0],
        }
        df_coherence_npmi = df_coherence_npmi.append(new_row, ignore_index=True)

    df_coherence_npmi[["No. topics"]] = df_coherence_npmi[["No. topics"]].astype(int)

    df_coherence_npmi["Configuration"] = pd.Categorical(
        df_coherence_npmi["Configuration"],
        categories=df_coherence_npmi["Configuration"].tolist(),
        ordered=True,
    )

    # Load data about uniques
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

    # Join all statistics together
    df_pnu = df_retrieval_metrics.merge(
        df_coherence_npmi, on="Configuration", how="outer"
    ).merge(df_uniques, on="Configuration", how="outer")

    selected_columns = [
        "Configuration",
        "No. topics_x",
        "Purity average",
        "Purity standard deviation",
        "NPMI average",
        "Uniques average",
        "Uniques standard deviation",
    ]
    df_pnu_selected = df_pnu[selected_columns]
    df_pnu_rows_selected = df_pnu_selected.iloc[30:44]
    print(df_pnu_rows_selected)

    return 0


if __name__ == "__main__":
    print("helper_selector_npmi_uniques_purity.py is being run directly")
else:
    print(
        "helper_selector_npmi_uniques_purity.py is being imported into another module"
    )
