"""
Factory class for creating instances of TopicMetrics subclasses.
"""

from typing import Optional, Type

from model.topic.btm_topic_metrics import BTMTopicMetrics
from model.topic.etm_topic_metrics import ETMTopicMetrics
from model.topic.lda_topic_metrics import LdaTopicMetrics
from model.topic.stm_topic_metrics import STMTopicMetrics
from model.topic.topic_metrics import TopicMetrics


class TopicMetricsFactory:
    """
    Factory class for creating instances of TopicMetrics subclasses.

    This factory class provides a convenient way to create instances of concrete subclasses
    of TopicMetrics based on the specified model_type.

    Attributes:
        _metric_classes (dict): A dictionary mapping supported model types to their
            corresponding metric classes.
    """

    _metric_classes: dict = {
        "STM": STMTopicMetrics,
        "lda_auto": LdaTopicMetrics,
        "btm": BTMTopicMetrics,
        "etm": ETMTopicMetrics,
    }

    @staticmethod
    def get_metric(
        model_type, number_of_topics, model, endpoint, word_number=10
    ) -> TopicMetrics:
        """
        Get an instance of a concrete TopicMetrics subclass based on the provided model_type.

        This method returns an instance of a concrete subclass of TopicMetrics based on
        the provided model_type. The instantiated class is responsible for calculating topic-related
        metrics based on the given model.

        Args:
            model_type (str): The type of the model for which to obtain the metrics.
                            Supported types include "STM", "lda_auto", "btm", and "etm".
            number_of_topics (int): The number of topics in the model.
            model: The model instance for which metrics will be calculated.
            endpoint (str): The endpoint for the model.
            word_number (int, optional): The number of words per topic used for metric calculations.
                        Default is 10.

        Returns:
            TopicMetrics: An instance of a concrete subclass of TopicMetrics, specific to the
                    provided model_type.

        Raises:
            Exception: If the specified model_type does not correspond to
                        any supported metrics class.
        """

        # We must take into account etm model that has a format etm_1, ...
        if "_" in model_type:
            model_type = model_type.split("_")[0]

        metric_class: Optional[
            Type[TopicMetrics]
        ] = TopicMetricsFactory._metric_classes.get(model_type)

        if metric_class:
            return metric_class(number_of_topics, model, endpoint, word_number)

        raise Exception(f"Topic metrics not implemented for {model_type}")
