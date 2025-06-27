"""
Memory configuration presets and utilities.

Provides pre-configured setups for different deployment scenarios:
- Local-only (no API keys required)
- Cloud-based (requires API keys)
- Hybrid configurations
"""

import os
from typing import Dict, Any, Optional


class MemoryConfig:
    """Configuration builder for AgentMemory instances."""
    
    @staticmethod
    def local_only(
        chroma_path: str = "./chroma_db",
        collection_name: str = "mem0_memory"
    ) -> Dict[str, Any]:
        """
        Fully local configuration using local models.
        No API keys required.
        
        Requires:
        - Ollama running locally
        - sentence-transformers installed
        
        Args:
            chroma_path: Path for ChromaDB storage
            collection_name: Name of the ChromaDB collection
            
        Returns:
            Configuration dict for local deployment
        """
        return {
            "vector_store": {
                "provider": "chroma",
                "config": {
                    "collection_name": collection_name,
                    "path": chroma_path,
                }
            },
            "llm": {
                "provider": "ollama",
                "config": {
                    "model": "llama3",
                    "temperature": 0.1,
                }
            },
            "embedder": {
                "provider": "openai",
                "config": {
                    "model": "text-embedding-3-small"
                }
            }
        }
    
    @staticmethod
    def openai_config(
        chroma_path: str = "./chroma_db",
        collection_name: str = "mem0_memory",
        llm_model: str = "gpt-4o-mini",
        embedding_model: str = "text-embedding-3-small"
    ) -> Dict[str, Any]:
        """
        OpenAI-based configuration.
        Requires OPENAI_API_KEY environment variable or api_key parameter.
        
        Args:
            chroma_path: Path for ChromaDB storage
            collection_name: Name of the ChromaDB collection
            llm_model: OpenAI model for LLM operations
            embedding_model: OpenAI model for embeddings
            
        Returns:
            Configuration dict for OpenAI deployment
        """
        return {
            "vector_store": {
                "provider": "chroma",
                "config": {
                    "collection_name": collection_name,
                    "path": chroma_path,
                }
            },
            "llm": {
                "provider": "openai",
                "config": {
                    "model": llm_model,
                    "temperature": 0.1,
                }
            },
            "embedder": {
                "provider": "openai",
                "config": {
                    "model": embedding_model
                }
            }
        }
    
    @staticmethod
    def google_config(
        chroma_path: str = "./chroma_db",
        collection_name: str = "mem0_memory",
        llm_model: str = "gemini-1.5-pro",
        embedding_model: str = "text-embedding-3-small"
    ) -> Dict[str, Any]:
        """
        Google + OpenAI hybrid configuration.
        Requires GOOGLE_API_KEY and OPENAI_API_KEY environment variables.
        
        Args:
            chroma_path: Path for ChromaDB storage
            collection_name: Name of the ChromaDB collection
            llm_model: Google model for LLM operations
            embedding_model: OpenAI model for embeddings
            
        Returns:
            Configuration dict for Google + OpenAI deployment
        """
        return {
            "vector_store": {
                "provider": "chroma",
                "config": {
                    "collection_name": collection_name,
                    "path": chroma_path,
                }
            },
            "llm": {
                "provider": "openai",
                "config": {
                    "model": "gpt-4o-mini",
                    "temperature": 0.1,
                }
            },
            "embedder": {
                "provider": "openai",
                "config": {
                    "model": embedding_model
                }
            }
        }
    
    @staticmethod
    def huggingface_config(
        chroma_path: str = "./chroma_db",
        collection_name: str = "mem0_memory",
        llm_model: str = "microsoft/DialoGPT-medium",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    ) -> Dict[str, Any]:
        """
        HuggingFace-based configuration.
        Requires HUGGINGFACE_API_KEY environment variable.
        
        Args:
            chroma_path: Path for ChromaDB storage
            collection_name: Name of the ChromaDB collection
            llm_model: HuggingFace model for LLM operations
            embedding_model: HuggingFace model for embeddings
            
        Returns:
            Configuration dict for HuggingFace deployment
        """
        return {
            "vector_store": {
                "provider": "chroma",
                "config": {
                    "collection_name": collection_name,
                    "path": chroma_path,
                }
            },
            "llm": {
                "provider": "huggingface",
                "config": {
                    "model": llm_model,
                    "temperature": 0.1,
                }
            },
            "embedder": {
                "provider": "huggingface",
                "config": {
                    "model": embedding_model
                }
            }
        }
    
    @staticmethod
    def anthropic_config(
        chroma_path: str = "./chroma_db",
        collection_name: str = "mem0_memory",
        llm_model: str = "claude-3-haiku-20240307",
        embedding_model: str = "text-embedding-3-small"
    ) -> Dict[str, Any]:
        """
        Anthropic + OpenAI hybrid configuration.
        Requires ANTHROPIC_API_KEY and OPENAI_API_KEY environment variables.
        
        Args:
            chroma_path: Path for ChromaDB storage
            collection_name: Name of the ChromaDB collection
            llm_model: Anthropic model for LLM operations
            embedding_model: OpenAI model for embeddings (Anthropic doesn't have embeddings)
            
        Returns:
            Configuration dict for Anthropic + OpenAI deployment
        """
        return {
            "vector_store": {
                "provider": "chroma",
                "config": {
                    "collection_name": collection_name,
                    "path": chroma_path,
                }
            },
            "llm": {
                "provider": "anthropic",
                "config": {
                    "model": llm_model,
                    "temperature": 0.1,
                }
            },
            "embedder": {
                "provider": "openai",
                "config": {
                    "model": embedding_model
                }
            }
        }
    
    @staticmethod
    def load_from_file(config_path: str) -> Dict[str, Any]:
        """
        Load configuration from a JSON file.
        
        Args:
            config_path: Path to the configuration file
            
        Returns:
            Configuration dict
        """
        import json
        
        with open(config_path, 'r') as f:
            return json.load(f)
    
    @staticmethod
    def save_to_file(config: Dict[str, Any], config_path: str) -> None:
        """
        Save configuration to a JSON file.
        
        Args:
            config: Configuration dict to save
            config_path: Path where to save the configuration
        """
        import json
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
    
    @staticmethod
    def detect_available_providers() -> Dict[str, bool]:
        """
        Detect which providers are available based on environment variables.
        
        Returns:
            Dict indicating which providers have required credentials
        """
        return {
            "openai": bool(os.environ.get("OPENAI_API_KEY")),
            "google": bool(os.environ.get("GOOGLE_API_KEY")),
            "anthropic": bool(os.environ.get("ANTHROPIC_API_KEY")),
            "huggingface": bool(os.environ.get("HUGGINGFACE_API_KEY")),
            "ollama": True,  # Assume locally available
            "sentence_transformers": True,  # Assume locally available
        }
    
    @staticmethod
    def get_recommended_config() -> Dict[str, Any]:
        """
        Get a recommended configuration based on available providers.
        
        Returns:
            Best available configuration
        """
        available = MemoryConfig.detect_available_providers()
        
        if available["openai"]:
            return MemoryConfig.openai_config()
        elif available["anthropic"] and available["openai"]:
            return MemoryConfig.anthropic_config()
        elif available["google"]:
            return MemoryConfig.google_config()
        elif available["huggingface"]:
            return MemoryConfig.huggingface_config()
        else:
            return MemoryConfig.local_only()


# Convenience functions for quick setup
def get_local_config(**kwargs) -> Dict[str, Any]:
    """Get local-only configuration (no API keys required)."""
    return MemoryConfig.local_only(**kwargs)


def get_openai_config(**kwargs) -> Dict[str, Any]:
    """Get OpenAI configuration (requires OPENAI_API_KEY)."""
    return MemoryConfig.openai_config(**kwargs)


def get_google_config(**kwargs) -> Dict[str, Any]:
    """Get Google + OpenAI configuration (requires both API keys)."""
    return MemoryConfig.google_config(**kwargs)


def get_anthropic_config(**kwargs) -> Dict[str, Any]:
    """Get Anthropic + OpenAI configuration (requires both API keys)."""
    return MemoryConfig.anthropic_config(**kwargs)


def get_recommended_config() -> Dict[str, Any]:
    """Get recommended configuration based on available API keys."""
    return MemoryConfig.get_recommended_config()


def get_test_config(collection_name: str = "test_memory") -> Dict[str, Any]:
    """
    Test configuration that works without real API keys.
    Uses in-memory ChromaDB to avoid conflicts.
    
    Args:
        collection_name: Unique collection name for the test
        
    Returns:
        Configuration dict for testing
    """
    return {
        "vector_store": {
            "provider": "chroma",
            "config": {
                "collection_name": collection_name,
                # No path specified = in-memory storage
            }
        },
        "llm": {
            "provider": "openai",
            "config": {
                "model": "gpt-4o-mini",
                "temperature": 0.1,
            }
        },
        "embedder": {
            "provider": "openai",
            "config": {
                "model": "text-embedding-3-small"
            }
        }
    }