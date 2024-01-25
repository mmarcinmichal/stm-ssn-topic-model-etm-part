""" Get best and worst topics"""
import pickle as pkl
from embedded_topic_model.models.etm import ETM

# Updated list of datasets and their corresponding model names
dataset_model_mapping = {
    "20news": ["etm_43_20_0"],
    "bbc": ["etm_59_20_0"],
    "ag": ["etm_39_20_0"],
}

for dataset, model_names in dataset_model_mapping.items():
    OUTPUT_PATH = f"model-output-data/{dataset}"

    for model_name in model_names:
        with open(f"{OUTPUT_PATH}/{model_name}", "rb") as file:
            loaded_object = pkl.load(file)

        if isinstance(loaded_object, ETM):
            etm_instance = loaded_object
            topics = etm_instance.get_topics()
            formatted_topics = [" ".join(topic) for topic in topics]
            three_best = formatted_topics[:3]
            three_worst = formatted_topics[-3:]

            print(f"Dataset: {dataset}, Model: {model_name}")
            print("Top 3 topics:")
            for topic in three_best:
                print(topic)
            print("Bottom 3 topics:")
            for topic in three_worst:
                print(topic)
        else:
            print(f"Dataset: {dataset}, Model: {model_name}")
            print("The loaded data is not an instance of ETM.")
