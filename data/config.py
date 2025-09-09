"""
Configuration module for video question-answer generation.

This module handles configuration settings including API keys,
file paths, and processing parameters.
"""

import os
from typing import Optional


class Config:
    """Configuration class for the QA generation system."""
    
    # API Configuration
    OPENAI_API_KEY: Optional[str] = os.getenv('OPENAI_API_KEY')
    
    # File paths
    BASE_DATA_DIR: str = os.getenv('BASE_DATA_DIR', '/pasteur/u/ruili')
    COGVLM_DIR: str = os.path.join(BASE_DATA_DIR, 'CogVLM2')
    DATA_CURATION_DIR: str = os.path.join(BASE_DATA_DIR, 'data_curation')
    PROMPT_DIR: str = 'prompt'
    
    # Processing parameters
    DEFAULT_MAX_TOKENS: int = 2000
    DEFAULT_TEMPERATURE_GENERATION: float = 0.8
    DEFAULT_TEMPERATURE_FILTERING: float = 0.0
    DEFAULT_TIMEOUT: int = 180
    DEFAULT_NUM_PROCESSES: int = 200
    DEFAULT_MAX_RETRIES: int = 2
    DEFAULT_FRAME_RANGE: int = 4
    TOTAL_FRAMES: int = 32
    
    # Dataset configuration
    CAPTION_NAME: str = "caption_new_frames_4_total"
    CURRENT_TYPE: str = "causual"
    
    @classmethod
    def get_openai_api_key(cls) -> str:
        """
        Get OpenAI API key with proper error handling.
        
        Returns:
            API key string
            
        Raises:
            ValueError: If API key is not configured
        """
        if not cls.OPENAI_API_KEY:
            raise ValueError(
                "OpenAI API key not found. Please set the OPENAI_API_KEY environment variable."
            )
        return cls.OPENAI_API_KEY
    
    @classmethod
    def get_input_path(cls, filename: str) -> str:
        """Get full path for input data file."""
        return os.path.join(cls.COGVLM_DIR, filename)
    
    @classmethod
    def get_output_path(cls, filename: str) -> str:
        """Get full path for output data file."""
        return os.path.join(cls.DATA_CURATION_DIR, filename)
    
    @classmethod
    def get_prompt_path(cls, filename: str) -> str:
        """Get full path for prompt template file."""
        return os.path.join(cls.PROMPT_DIR, filename)
    
    @classmethod
    def get_output_name(cls) -> str:
        """Generate output filename based on configuration."""
        return cls.CAPTION_NAME.replace("caption", "qa") + f"_{cls.CURRENT_TYPE}_new"

