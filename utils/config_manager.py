# utils/config_manager.py

import configparser
import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)

class ConfigManager:
    """
    A production‑level configuration manager that loads settings from a config file.

    This class provides methods to retrieve configuration values in a type‑safe manner,
    as well as to obtain an entire configuration section as a dictionary.
    """
    
    def __init__(self, config_file: str = "config.ini") -> None:
        """
        Initialize the ConfigManager by loading the configuration from the specified file.
        
        Args:
            config_file (str): Path to the configuration file (default is "config.ini").
        """
        self.config_file: str = config_file
        self.config: configparser.ConfigParser = configparser.ConfigParser()
        self._load_config()
    
    def _load_config(self) -> None:
        """
        Load the configuration from the file.
        
        Raises:
            FileNotFoundError: If the configuration file is not found.
            Exception: Propagates any exception encountered during reading.
        """
        try:
            read_files = self.config.read(self.config_file)
            if not read_files:
                logger.error(f"No configuration file found at '{self.config_file}'.")
                raise FileNotFoundError(f"Configuration file {self.config_file} not found.")
            logger.info(f"Configuration successfully loaded from {self.config_file}.")
        except Exception as e:
            logger.error(f"Error loading configuration from {self.config_file}: {e}")
            raise
    
    def get(self, section: str, key: str, fallback: Any = None) -> Any:
        """
        Retrieve a configuration value as a string from a given section, with an optional fallback.
        
        Args:
            section (str): The configuration section.
            key (str): The key within the section.
            fallback (Any): The fallback value if the key is not found.
        
        Returns:
            Any: The value associated with the key, or the fallback if not available.
        """
        try:
            return self.config.get(section, key, fallback=fallback)
        except Exception as e:
            logger.error(f"Error retrieving key '{key}' from section '{section}': {e}")
            return fallback
    
    def get_int(self, section: str, key: str, fallback: int = 0) -> int:
        """
        Retrieve an integer configuration value.
        
        Args:
            section (str): The configuration section.
            key (str): The key within the section.
            fallback (int): The fallback value if the key is not found or conversion fails.
        
        Returns:
            int: The integer value.
        """
        try:
            return self.config.getint(section, key, fallback=fallback)
        except Exception as e:
            logger.error(f"Error retrieving int key '{key}' from section '{section}': {e}")
            return fallback
    
    def get_float(self, section: str, key: str, fallback: float = 0.0) -> float:
        """
        Retrieve a float configuration value.
        
        Args:
            section (str): The configuration section.
            key (str): The key within the section.
            fallback (float): The fallback value if the key is not found or conversion fails.
        
        Returns:
            float: The float value.
        """
        try:
            return self.config.getfloat(section, key, fallback=fallback)
        except Exception as e:
            logger.error(f"Error retrieving float key '{key}' from section '{section}': {e}")
            return fallback
    
    def get_boolean(self, section: str, key: str, fallback: bool = False) -> bool:
        """
        Retrieve a boolean configuration value.
        
        Args:
            section (str): The configuration section.
            key (str): The key within the section.
            fallback (bool): The fallback value if the key is not found or conversion fails.
        
        Returns:
            bool: The boolean value.
        """
        try:
            return self.config.getboolean(section, key, fallback=fallback)
        except Exception as e:
            logger.error(f"Error retrieving boolean key '{key}' from section '{section}': {e}")
            return fallback
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """
        Retrieve all configuration values for the specified section as a dictionary.
        
        Args:
            section (str): The configuration section.
        
        Returns:
            Dict[str, Any]: A dictionary with all keys and values of the section.
                            Returns an empty dictionary if the section does not exist.
        """
        if section not in self.config:
            logger.warning(f"Section '{section}' not found in config file {self.config_file}.")
            return {}
        return dict(self.config[section])