from topic_model_builder_full_configurations_ETM import (
    configurations as EtmModelsConfigurations,
)


class ETMConfigurationContext:
    """
    ETM has different structure of configuration.
    So we need make some additional operations.
    After execution of ETM configuration all is restored to previous state
    if after etm is executed other model, i.e. TOPICS_RANGE_NUMBER global list is restored.
    """

    def __init__(self, configuration_model_name, topics_range_number):
        self.configuration_model_name = configuration_model_name
        self.topics_range_number = topics_range_number
        self.topics_range_number_copy = None

    def __enter__(self):
        if self.configuration_model_name.startswith("etm_"):
            self.topics_range_number_copy = self.topics_range_number.copy()
            configuration = EtmModelsConfigurations.get(
                self.configuration_model_name, {}
            )
            num_topics = configuration.get("num_topics")
            self.topics_range_number.clear()
            self.topics_range_number.append(num_topics)

    def __exit__(self, exc_type, exc_value, traceback):
        if self.topics_range_number_copy:
            self.topics_range_number.clear()
            self.topics_range_number.extend(self.topics_range_number_copy)
