"""
Unit tests for AgentMemory and MemoryConfig.
"""
import os
import json
import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List

from ..memory.agent_memory import AgentMemory, create_agent_memory
from ..memory.memory_config import MemoryConfig, get_test_config, get_recommended_config


class TestMemoryConfig:
    """Test cases for MemoryConfig class"""
    
    def test_local_only_config(self):
        """Test local-only configuration generation"""
        config = MemoryConfig.local_only()
        
        assert config["vector_store"]["provider"] == "chroma"
        assert config["llm"]["provider"] == "ollama"
        assert config["embedder"]["provider"] == "openai"
        assert "collection_name" in config["vector_store"]["config"]
        assert "path" in config["vector_store"]["config"]
    
    def test_local_only_config_custom_params(self):
        """Test local-only configuration with custom parameters"""
        custom_path = "/custom/path"
        custom_collection = "custom_collection"
        
        config = MemoryConfig.local_only(
            chroma_path=custom_path,
            collection_name=custom_collection
        )
        
        assert config["vector_store"]["config"]["path"] == custom_path
        assert config["vector_store"]["config"]["collection_name"] == custom_collection
    
    def test_openai_config(self):
        """Test OpenAI configuration generation"""
        config = MemoryConfig.openai_config()
        
        assert config["vector_store"]["provider"] == "chroma"
        assert config["llm"]["provider"] == "openai"
        assert config["embedder"]["provider"] == "openai"
        assert config["llm"]["config"]["model"] == "gpt-4o-mini"
        assert config["embedder"]["config"]["model"] == "text-embedding-3-small"
    
    def test_google_config(self):
        """Test Google configuration generation"""
        config = MemoryConfig.google_config()
        
        assert config["vector_store"]["provider"] == "chroma"
        assert config["llm"]["provider"] == "openai"  # Uses OpenAI LLM in the code
        assert config["embedder"]["provider"] == "openai"
    
    def test_anthropic_config(self):
        """Test Anthropic configuration generation"""
        config = MemoryConfig.anthropic_config()
        
        assert config["vector_store"]["provider"] == "chroma"
        assert config["llm"]["provider"] == "anthropic"
        assert config["embedder"]["provider"] == "openai"
        assert config["llm"]["config"]["model"] == "claude-3-haiku-20240307"
    
    def test_huggingface_config(self):
        """Test HuggingFace configuration generation"""
        config = MemoryConfig.huggingface_config()
        
        assert config["vector_store"]["provider"] == "chroma"
        assert config["llm"]["provider"] == "huggingface"
        assert config["embedder"]["provider"] == "huggingface"
    
    def test_detect_available_providers_no_keys(self):
        """Test provider detection with no API keys"""
        with patch.dict(os.environ, {}, clear=True):
            providers = MemoryConfig.detect_available_providers()
            
            assert providers["openai"] is False
            assert providers["google"] is False
            assert providers["anthropic"] is False
            assert providers["huggingface"] is False
            assert providers["ollama"] is True
            assert providers["sentence_transformers"] is True
    
    def test_detect_available_providers_with_keys(self):
        """Test provider detection with API keys"""
        env_vars = {
            "OPENAI_API_KEY": "test-key",
            "GOOGLE_API_KEY": "test-key",
            "ANTHROPIC_API_KEY": "test-key",
            "HUGGINGFACE_API_KEY": "test-key"
        }
        
        with patch.dict(os.environ, env_vars):
            providers = MemoryConfig.detect_available_providers()
            
            assert providers["openai"] is True
            assert providers["google"] is True
            assert providers["anthropic"] is True
            assert providers["huggingface"] is True
    
    def test_get_recommended_config_with_openai(self):
        """Test recommended config when OpenAI key is available"""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=True):
            config = MemoryConfig.get_recommended_config()
            assert config["llm"]["provider"] == "openai"
    
    def test_get_recommended_config_no_keys(self):
        """Test recommended config when no API keys are available"""
        with patch.dict(os.environ, {}, clear=True):
            config = MemoryConfig.get_recommended_config()
            assert config["llm"]["provider"] == "ollama"
    
    def test_get_test_config(self):
        """Test test configuration generation"""
        config = get_test_config()
        
        assert config["vector_store"]["provider"] == "chroma"
        assert config["llm"]["provider"] == "openai"
        assert config["embedder"]["provider"] == "openai"
        # In-memory ChromaDB should not have a path
        assert "path" not in config["vector_store"]["config"]
    
    def test_save_and_load_config(self, tmp_path):
        """Test saving and loading configuration to/from file"""
        config = MemoryConfig.openai_config()
        config_file = tmp_path / "test_config.json"
        
        # Save config
        MemoryConfig.save_to_file(config, str(config_file))
        assert config_file.exists()
        
        # Load config
        loaded_config = MemoryConfig.load_from_file(str(config_file))
        assert loaded_config == config


class TestAgentMemory:
    """Test cases for AgentMemory class"""
    
    def setup_method(self):
        """Set up test fixtures"""
        # Mock the Mem0 Memory class to avoid requiring real API keys
        self.mock_memory = Mock()
        self.mock_memory.add.return_value = {
            "results": [{"id": "mem_123"}, {"id": "mem_456"}]
        }
        self.mock_memory.search.return_value = {
            "results": [
                {
                    "id": "mem_123",
                    "memory": "User likes pizza",
                    "score": 0.95,
                    "metadata": {"session_id": "session_1"}
                }
            ]
        }
        self.mock_memory.get_all.return_value = {
            "results": [
                {
                    "id": "mem_123",
                    "memory": "User likes pizza",
                    "metadata": {"session_id": "session_1", "agent_id": "agent_1"}
                },
                {
                    "id": "mem_456", 
                    "memory": "User prefers tea",
                    "metadata": {"session_id": "session_2"}
                }
            ]
        }
        self.mock_memory.update.return_value = {"status": "success"}
        self.mock_memory.delete.return_value = None
        
        # Use test configuration that doesn't require real API keys
        self.test_config = get_test_config("test_collection")
    
    @patch('jentic_agents.memory.agent_memory.Memory')
    def test_init_with_test_config(self, mock_memory_class):
        """Test AgentMemory initialization with test configuration"""
        mock_memory_class.from_config.return_value = self.mock_memory
        
        memory = AgentMemory(config=self.test_config)
        
        assert memory.config == self.test_config
        assert memory.api_key is None
        mock_memory_class.from_config.assert_called_once_with(self.test_config)
    
    @patch('jentic_agents.memory.agent_memory.Memory')
    def test_init_with_api_key(self, mock_memory_class):
        """Test AgentMemory initialization with API key"""
        mock_memory_class.from_config.return_value = self.mock_memory
        
        memory = AgentMemory(config=self.test_config, api_key="test-key")
        
        assert memory.api_key == "test-key"
        # Should set environment variables for OpenAI
        assert os.environ.get("OPENAI_API_KEY") == "test-key"
    
    @patch('jentic_agents.memory.agent_memory.Memory')
    def test_init_use_recommended(self, mock_memory_class):
        """Test AgentMemory initialization with recommended config"""
        mock_memory_class.from_config.return_value = self.mock_memory
        
        with patch.dict(os.environ, {}, clear=True):
            memory = AgentMemory(use_recommended=True)
            
            # Should use local config when no API keys available
            assert memory.config["llm"]["provider"] == "ollama"
    
    @patch('jentic_agents.memory.agent_memory.Memory')
    def test_add_single_message(self, mock_memory_class):
        """Test adding a single message"""
        mock_memory_class.from_config.return_value = self.mock_memory
        memory = AgentMemory(config=self.test_config)
        
        result = memory.add("Test message", user_id="user_1")
        
        assert result == ["mem_123", "mem_456"]
        self.mock_memory.add.assert_called_once_with(
            "Test message",
            user_id="user_1",
            metadata={}
        )
    
    @patch('jentic_agents.memory.agent_memory.Memory')
    def test_add_with_metadata(self, mock_memory_class):
        """Test adding message with metadata"""
        mock_memory_class.from_config.return_value = self.mock_memory
        memory = AgentMemory(config=self.test_config)
        
        result = memory.add(
            "Test message",
            user_id="user_1",
            agent_id="agent_1",
            session_id="session_1",
            metadata={"custom": "value"}
        )
        
        expected_metadata = {
            "agent_id": "agent_1",
            "session_id": "session_1",
            "custom": "value"
        }
        self.mock_memory.add.assert_called_once_with(
            "Test message",
            user_id="user_1",
            metadata=expected_metadata
        )
    
    @patch('jentic_agents.memory.agent_memory.Memory')
    def test_search_basic(self, mock_memory_class):
        """Test basic memory search"""
        mock_memory_class.from_config.return_value = self.mock_memory
        memory = AgentMemory(config=self.test_config)
        
        result = memory.search("pizza", user_id="user_1")
        
        assert len(result["results"]) == 1
        assert result["results"][0]["memory"] == "User likes pizza"
        self.mock_memory.search.assert_called_once_with(
            query="pizza",
            user_id="user_1",
            limit=5
        )
    
    @patch('jentic_agents.memory.agent_memory.Memory')
    def test_search_with_filters(self, mock_memory_class):
        """Test memory search with session/agent filters"""
        mock_memory_class.from_config.return_value = self.mock_memory
        memory = AgentMemory(config=self.test_config)
        
        # Mock search to return multiple results for filtering
        self.mock_memory.search.return_value = {
            "results": [
                {
                    "id": "mem_123",
                    "memory": "User likes pizza",
                    "score": 0.95,
                    "metadata": {"session_id": "session_1", "agent_id": "agent_1"}
                },
                {
                    "id": "mem_456",
                    "memory": "User prefers tea", 
                    "score": 0.90,
                    "metadata": {"session_id": "session_2", "agent_id": "agent_1"}
                }
            ]
        }
        
        result = memory.search(
            "food",
            user_id="user_1",
            session_id="session_1",
            agent_id="agent_1"
        )
        
        # Should filter to only session_1 results
        assert len(result["results"]) == 1
        assert result["results"][0]["metadata"]["session_id"] == "session_1"
    
    @patch('jentic_agents.memory.agent_memory.Memory')
    def test_get_all_basic(self, mock_memory_class):
        """Test getting all memories"""
        mock_memory_class.from_config.return_value = self.mock_memory
        memory = AgentMemory(config=self.test_config)
        
        result = memory.get_all(user_id="user_1")
        
        assert len(result) == 2
        self.mock_memory.get_all.assert_called_once_with(user_id="user_1", limit=100)
    
    @patch('jentic_agents.memory.agent_memory.Memory')
    def test_get_all_with_filters(self, mock_memory_class):
        """Test getting all memories with filters"""
        mock_memory_class.from_config.return_value = self.mock_memory
        memory = AgentMemory(config=self.test_config)
        
        result = memory.get_all(user_id="user_1", agent_id="agent_1")
        
        # Should filter to only agent_1 results
        assert len(result) == 1
        assert result[0]["metadata"]["agent_id"] == "agent_1"
    
    @patch('jentic_agents.memory.agent_memory.Memory')
    def test_update_memory(self, mock_memory_class):
        """Test updating a memory"""
        mock_memory_class.from_config.return_value = self.mock_memory
        memory = AgentMemory(config=self.test_config)
        
        result = memory.update("mem_123", "Updated content")
        
        assert result == {"status": "success"}
        self.mock_memory.update.assert_called_once_with(
            memory_id="mem_123",
            data="Updated content"
        )
    
    @patch('jentic_agents.memory.agent_memory.Memory')
    def test_delete_memory_success(self, mock_memory_class):
        """Test successful memory deletion"""
        mock_memory_class.from_config.return_value = self.mock_memory
        memory = AgentMemory(config=self.test_config)
        
        result = memory.delete_memory("mem_123")
        
        assert result is True
        self.mock_memory.delete.assert_called_once_with(memory_id="mem_123")
    
    @patch('jentic_agents.memory.agent_memory.Memory')
    def test_delete_memory_failure(self, mock_memory_class):
        """Test memory deletion failure"""
        mock_memory_class.from_config.return_value = self.mock_memory
        memory = AgentMemory(config=self.test_config)
        
        # Mock delete to raise an exception
        self.mock_memory.delete.side_effect = Exception("Delete failed")
        
        result = memory.delete_memory("mem_123")
        
        assert result is False
    
    @patch('jentic_agents.memory.agent_memory.Memory')
    def test_delete_all(self, mock_memory_class):
        """Test deleting all memories for a user"""
        mock_memory_class.from_config.return_value = self.mock_memory
        memory = AgentMemory(config=self.test_config)
        
        # Mock successful deletions
        with patch.object(memory, 'delete_memory', return_value=True) as mock_delete:
            result = memory.delete_all("user_1")
            
            assert result == 2  # Should delete 2 memories
            assert mock_delete.call_count == 2
    
    @patch('jentic_agents.memory.agent_memory.Memory')
    def test_get_memory_usage(self, mock_memory_class):
        """Test getting memory usage statistics"""
        mock_memory_class.from_config.return_value = self.mock_memory
        memory = AgentMemory(config=self.test_config)
        
        usage = memory.get_memory_usage("user_1")
        
        assert usage["user_id"] == "user_1"
        assert usage["total_memories"] == 2
        assert usage["unique_sessions"] == 2
        assert usage["unique_agents"] == 1
        assert "session_1" in usage["sessions"]
        assert "session_2" in usage["sessions"]
        assert "agent_1" in usage["agents"]
    
    @patch('jentic_agents.memory.agent_memory.Memory')
    def test_get_context_for_chat(self, mock_memory_class):
        """Test getting context for chat"""
        mock_memory_class.from_config.return_value = self.mock_memory
        memory = AgentMemory(config=self.test_config)
        
        context = memory.get_context_for_chat("food", "user_1")
        
        assert "Relevant memories:" in context
        assert "User likes pizza" in context
        assert "relevance:" in context
    
    @patch('jentic_agents.memory.agent_memory.Memory')
    def test_get_context_for_chat_no_results(self, mock_memory_class):
        """Test getting context when no memories found"""
        mock_memory_class.from_config.return_value = self.mock_memory
        memory = AgentMemory(config=self.test_config)
        
        # Mock empty search results
        self.mock_memory.search.return_value = {"results": []}
        
        context = memory.get_context_for_chat("unknown", "user_1")
        
        assert context == "No relevant memories found."
    
    @patch('jentic_agents.memory.agent_memory.Memory')
    def test_health_check_success(self, mock_memory_class):
        """Test successful health check"""
        mock_memory_class.from_config.return_value = self.mock_memory
        memory = AgentMemory(config=self.test_config)
        
        # Mock successful add and delete for health check
        with patch.object(memory, 'add', return_value=["test_id"]), \
             patch.object(memory, 'delete_memory', return_value=True):
            
            health = memory.health_check()
            
            assert health["status"] == "healthy"
            assert "mem0_config" in health
            assert "timestamp" in health
    
    @patch('jentic_agents.memory.agent_memory.Memory')
    def test_health_check_failure(self, mock_memory_class):
        """Test health check failure"""
        mock_memory_class.from_config.return_value = self.mock_memory
        memory = AgentMemory(config=self.test_config)
        
        # Mock add to raise an exception
        with patch.object(memory, 'add', side_effect=Exception("Health check failed")):
            health = memory.health_check()
            
            assert health["status"] == "unhealthy"
            assert len(health["errors"]) > 0
    
    @patch('jentic_agents.memory.agent_memory.Memory')
    def test_base_memory_interface_store(self, mock_memory_class):
        """Test BaseMemory interface store method"""
        mock_memory_class.from_config.return_value = self.mock_memory
        memory = AgentMemory(config=self.test_config)
        
        memory.store("key1", "value1")
        
        self.mock_memory.add.assert_called_once_with(
            "value1",
            user_id="key1",
            metadata={}
        )
    
    @patch('jentic_agents.memory.agent_memory.Memory')
    def test_base_memory_interface_retrieve(self, mock_memory_class):
        """Test BaseMemory interface retrieve method"""
        mock_memory_class.from_config.return_value = self.mock_memory
        memory = AgentMemory(config=self.test_config)
        
        # Mock get_all to return memory data
        with patch.object(memory, 'get_all', return_value=[{
            "id": "mem_123",
            "memory": "retrieved value"
        }]):
            result = memory.retrieve("key1")
            
            assert result == "retrieved value"
    
    @patch('jentic_agents.memory.agent_memory.Memory')
    def test_base_memory_interface_retrieve_empty(self, mock_memory_class):
        """Test BaseMemory interface retrieve method with no results"""
        mock_memory_class.from_config.return_value = self.mock_memory
        memory = AgentMemory(config=self.test_config)
        
        with patch.object(memory, 'get_all', return_value=[]):
            result = memory.retrieve("nonexistent")
            
            assert result is None
    
    @patch('jentic_agents.memory.agent_memory.Memory')
    def test_base_memory_interface_delete(self, mock_memory_class):
        """Test BaseMemory interface delete method"""
        mock_memory_class.from_config.return_value = self.mock_memory
        memory = AgentMemory(config=self.test_config)
        
        with patch.object(memory, 'delete_all', return_value=2):
            result = memory.delete("key1")
            
            assert result is True
    
    @patch('jentic_agents.memory.agent_memory.Memory')
    def test_base_memory_interface_keys(self, mock_memory_class):
        """Test BaseMemory interface keys method"""
        mock_memory_class.from_config.return_value = self.mock_memory
        memory = AgentMemory(config=self.test_config)
        
        # keys() method is not fully supported, should return empty list
        result = memory.keys()
        
        assert result == []


class TestCreateAgentMemory:
    """Test cases for create_agent_memory factory function"""
    
    @patch('jentic_agents.memory.agent_memory.AgentMemory')
    def test_create_local_config(self, mock_agent_memory):
        """Test creating AgentMemory with local configuration"""
        create_agent_memory(config_type="local")
        
        # Verify AgentMemory was called with local config
        args, kwargs = mock_agent_memory.call_args
        assert kwargs["config"]["llm"]["provider"] == "ollama"
    
    @patch('jentic_agents.memory.agent_memory.AgentMemory')
    def test_create_openai_config(self, mock_agent_memory):
        """Test creating AgentMemory with OpenAI configuration"""
        create_agent_memory(config_type="openai", api_key="test-key")
        
        args, kwargs = mock_agent_memory.call_args
        assert kwargs["config"]["llm"]["provider"] == "openai"
        assert kwargs["api_key"] == "test-key"
    
    @patch('jentic_agents.memory.agent_memory.AgentMemory')
    def test_create_recommended_config(self, mock_agent_memory):
        """Test creating AgentMemory with recommended configuration"""
        with patch.dict(os.environ, {}, clear=True):
            create_agent_memory(config_type="recommended")
            
            args, kwargs = mock_agent_memory.call_args
            assert kwargs["use_recommended"] is True
    
    @patch('jentic_agents.memory.agent_memory.AgentMemory')
    def test_create_with_custom_chroma_path(self, mock_agent_memory):
        """Test creating AgentMemory with custom ChromaDB path"""
        custom_path = "/custom/chroma/path"
        create_agent_memory(config_type="local", chroma_path=custom_path)
        
        args, kwargs = mock_agent_memory.call_args
        assert kwargs["config"]["vector_store"]["config"]["path"] == custom_path
    
    def test_create_with_platform_no_api_key(self):
        """Test creating AgentMemory with platform mode but no API key"""
        with pytest.raises(ValueError, match="API key required"):
            create_agent_memory(config_type="local", use_platform=True)
    
    def test_create_unknown_config_type(self):
        """Test creating AgentMemory with unknown configuration type"""
        with pytest.raises(ValueError, match="Unknown config_type"):
            create_agent_memory(config_type="unknown")
    
    @patch('jentic_agents.memory.agent_memory.AgentMemory')
    def test_create_auto_detect_api_key(self, mock_agent_memory):
        """Test auto-detection of API key from environment"""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "env-key"}):
            create_agent_memory(config_type="openai")
            
            args, kwargs = mock_agent_memory.call_args
            assert kwargs["api_key"] == "env-key" 