import os.path
import pickle as pkl

from embedded_topic_model.models.etm import ETM
from embedded_topic_model.utils import embedding

from model.datasets.dataset_loader import DatasetLoader
from model.evaluation.retrival_metrics import RetrivalMetrics
from model.preprocessing.dataset_etm_creator import create_etm_datasets
from model.preprocessing.dataset_interface import DatasetInterface

DATA_SET = "bbc"
FEATURE_LIMIT = 2000
ITER = 0
TOPIC_NBR = 20

DATA_SET_PATH = f"model-input-data/{DATA_SET}"
MODEL_PATH = f"model-output-data/{DATA_SET}"

data_set_name = f"{DATA_SET}_{FEATURE_LIMIT}"
data_set: DatasetInterface = DatasetLoader(
    DATA_SET, FEATURE_LIMIT, DATA_SET_PATH
).load_dataset()

texts = [" ".join(doc) for doc in data_set.train_tokens()]

# PREPROCESSING
vocabulary, train_dataset, test_dataset = create_etm_datasets(
    texts, min_df=0, max_df=100_000, train_size=len(texts), debug_mode=True
)

# Training word2vec embeddings
embeddings_mapping = embedding.create_word2vec_embedding_from_dataset(texts)

# INITIALIZING AND RUNNING MODEL
STEPS_RANGE_NUMBER = 5
SEEDS = [2019, 1, 7, 28, 517]

for N in [20, 30, 40]:
    for i in range(STEPS_RANGE_NUMBER):
        model = ETM(
            vocabulary,
            embeddings=embeddings_mapping,  # You can pass here the path to a word2vec file
            # or a KeyedVectors instance
            num_topics=N,
            epochs=300,
            debug_mode=True,
            train_embeddings=False,
            # Optional. If True, ETM will learn word embeddings jointly with topic
            # embeddings. By default, is False. If 'embeddings' argument is being passed,
            # this argument must not be True
            seed=SEEDS[i],
        )
        model.fit(train_dataset)
        model_name = f"etm_{N}_{i}"
        if not os.path.exists(MODEL_PATH):
            os.mkdir(MODEL_PATH)
        with open(os.path.join(MODEL_PATH, f"etm_{N}_{i}"), "wb") as file:
            pkl.dump(model, file)

        train_norm = model.get_document_topic_dist().cpu().numpy()

        clustering_met: RetrivalMetrics = RetrivalMetrics(
            model_name,
            N,
            train_norm,
            train_norm,
            data_set.train_labels(),
            data_set.test_labels(),
            data_set.categories(),
        )
        clustering_met.calculate_metrics()

        print("Purity ETM: ", clustering_met.purity)
        print("F-score ETM: ", clustering_met.classification_metrics.fscore)

        clustering_met.save(MODEL_PATH, f"{model_name}")
