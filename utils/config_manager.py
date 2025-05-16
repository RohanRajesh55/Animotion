import configparser
import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)

class ConfigManager:
    """
    A production‑level configuration manager that loads settings from a config file.
    
    This class provides methods to retrieve configuration values in a type‑safe manner,
    as well as the entire configuration for a specified section.
    """
    
    def __init__(self, config_file: str = "config.ini") -> None:
        """
        Initialize the ConfigManager by loading the configuration from the specified file.
        
        Args:
            config_file (str): Path to the configuration file (default "config.ini").
        """
        self.config_file: str = config_file
        self.config: configparser.ConfigParser = configparser.ConfigParser()
        self._load_config()
    
    def _load_config(self) -> None:
        """
        Load the configuration from the file.
        
        Raises:
            Exception: Propagated if the configuration file cannot be read.
        """
        try:
            read_files = self.config.read(self.config_file)
            if not read_files:
                logger.error(f"No configuration file found at '{self.config_file}'.")
                raise FileNotFoundError(f"Configuration file {self.config_file} not found.")
            logger.info(f"Configuration successfully loaded from {self.config_file}.")
        except Exception as e:
            logger.error(f"Error loading configuration from {self.config_file}: {e}")
            raise e
        
    def get(self, section: str, key: str, fallback: Any = None) -> Any:
        """
        Retrieve a configuration value from a given section with an optional fallback.
        
        Args:
            section (str): The configuration section.
            key (str): The configuration key within the section.
            fallback (Any): The fallback value if the key is not found.
        
        Returns:
            Any: The value associated with the key, or the fallback if not available.
        """
        try:
            return self.config.get(section, key, fallback=fallback)
        except Exception as e:
            logger.error(f"Error retrieving key '{key}' from section '{section}': {e}")
            return fallback
        
    def get_section(self, section: str) -> Dict[str, Any]:
        """
        Retrieve all configuration values for the specified section as a dictionary.
        
        Args:
            section (str): The configuration section.
        
        Returns:
            Dict[str, Any]: A dictionary containing all keys and values for the section.
                          If the section does not exist, an empty dictionary is returned.
        """
        if section not in self.config:
            logger.warning(f"Section '{section}' not found in config file {self.config_file}.")
            return {}
        return dict(self.config[section])