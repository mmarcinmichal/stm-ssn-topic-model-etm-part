import os

import numpy as np
import pandas as pd
from plotnine import ggplot, aes, geom_point, labs, theme, element_text

from model.configuration.etm_configuration_context import ETMConfigurationContext
from model.evaluation.retrival_metrics import RetrivalMetrics
from topic_model_builder_full_configurations_ETM import (
    configurations as EtmModelsConfigurations,
)

# Currently we have default datasets such as: 20news, ag, and bbc
DATA_SET = "bbc"

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


def main() -> int:
    f_score: dict = {m: [] for m in MODELS_RANG_NAMES}
    purity: dict = {m: [] for m in MODELS_RANG_NAMES}

    for i in range(STEPS_RANGE_NUMBER):
        for m in MODELS_RANG_NAMES:
            with ETMConfigurationContext(m, TOPICS_RANGE_NUMBER):
                print(f"Computation for range: {m} {TOPICS_RANGE_NUMBER}")
                for N in TOPICS_RANGE_NUMBER:
                    model_name = f"{m}_{N}_{i}_clustering_metrics"
                    print(f"Computation model_name: {model_name}")
                    M: RetrivalMetrics = RetrivalMetrics.load(OUTPUT_PATH, model_name)
                    f_score[m].append(M.classification_metrics.fscore)
                    print(model_name, M.purity)
                    purity[m].append(M.purity)

    df_retrieval_metrics = pd.DataFrame()

    for m in MODELS_RANG_NAMES:
        with ETMConfigurationContext(m, TOPICS_RANGE_NUMBER):
            for N in TOPICS_RANGE_NUMBER:
                average_f_score = np.average(f_score[m]) * 100
                std_dev_f_score = np.std(f_score[m]) * 100
                print(
                    f"F-score for {m} and {N} ({TOPICS_RANGE_NUMBER}):"
                    f"{average_f_score:.2f} ({std_dev_f_score:.2f}))"
                )
                average_purity_score = np.average(purity[m]) * 100
                std_dev_purity_score = np.std(purity[m]) * 100
                print(
                    f"Purity for {m} and {N} ({TOPICS_RANGE_NUMBER}):"
                    f"{average_purity_score:.2f} ({std_dev_purity_score:.2f}))"
                )

                new_row = {
                    "Configuration": m,
                    "No. topics": N,
                    "F-score average": average_f_score,
                    "F-score standard deviation": std_dev_f_score,
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

    # Create the ggplot line plot
    # %matplotlib inline
    scatter_plot = (
        ggplot(df_retrieval_metrics, aes(x="Configuration", y="F-score average"))
        + geom_point()
        + labs(x="Configuration", y="F-score average")
        + theme(axis_text_x=element_text(angle=90, hjust=1))
    )
    scatter_plot.save(os.path.join(OUTPUT_PATH, "etm_scatter_plot_f-score-avg.png"))

    scatter_plot = (
        ggplot(df_retrieval_metrics, aes(x="Configuration", y="Purity average"))
        + geom_point()
        + labs(x="Configuration", y="Purity average")
        + theme(axis_text_x=element_text(angle=90, hjust=1))
    )
    scatter_plot.save(os.path.join(OUTPUT_PATH, "etm_scatter_plot_purity-avg.png"))

    return 0


if __name__ == "__main__":
    print(
        "topic_based_information_retrieval_purity_evaluation.py is being run directly"
    )
else:
    print(
        "topic_based_information_retrieval_purity_evaluation.py is being imported into another module"
    )
