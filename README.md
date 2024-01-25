# Spiking Topic Model (STM) - Embedding Topic Model (ETM) experiment part

This repository is part of experiments from https://github.com/mioun/stm-ssn-topic-model. It has been separated due to the need for more compatibility and library conflicts to run the ETM model with the main branch of experiments (https://github.com/mioun/stm-ssn-topic-model).

Key experiment scripts:
- topic_model_builder_full_configurations_ETM.py: the script contains the complete test configuration (parameters grid search) and is used to build ETM models for each dataset. Uncommented configurations in configurations_20news, configurations_bbc, and configurations_ag represent the final selected configurations for deeper analysis.
- topic_based_information_retrieval_purity_evaluation.py: the script computes metrics such as F-score and purity for selected models with the option for visualization.
- topic_model_builder_ETM.py: the script contains simplified testing of a single selected configuration of the ETM model (calculating F-score and purity).
- topic_model_evaluator.py: the script contains the evaluation of topics created using the ETM model (calculating topic coherence measures using Palmetto).

Additional scripts:
- helper_get_bw_topics.py: Selection of the top and bottom topics from the ranking.
- helper_selector_npmi_uniques_purity.py: Auxiliary script for analyzing topic metrics.
