import yaml


class Config:
    """
    Configuration class for the agent.
    """

    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = self.load_config()

    def load_config(self) -> dict:
        """
        Loads the configuration from a YAML file.

        Returns:
            A dictionary containing the configuration.
        """
        with open(self.config_path) as f:
            return yaml.safe_load(f)

    def get(self, key: str, default=None):
        """
        Gets a configuration value.

        Args:
            key: The key of the configuration value.
            default: The default value to return if the key is not found.

        Returns:
            The configuration value.
        """
        keys = key.split(".")
        value = self.config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k, default)
            else:
                return default
        return value
