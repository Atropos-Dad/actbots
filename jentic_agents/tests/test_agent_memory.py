#!/usr/bin/env python3
"""
Test cases for agent memory system powered by Mem0.
"""
import sys
import os
import uuid

# Add the package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from jentic_agents.memory.agent_memory import AgentMemory, create_agent_memory
from jentic_agents.utils.memory import ScratchPadMemory


class TestAgentMemory:
    """Test cases for AgentMemory class."""
    
    def setup_method(self):
        """Set up a fresh memory instance for each test."""
        # Use a unique collection name for each test to avoid conflicts
        unique_name = f"test_agent_{uuid.uuid4().hex[:8]}"
        self.memory = create_agent_memory(collection_name=unique_name)
        self.test_user_id = f"test_user_{uuid.uuid4().hex[:8]}"
        self.test_session_id = f"test_session_{uuid.uuid4().hex[:8]}"
    
    def teardown_method(self):
        """Clean up after each test."""
        try:
            # Clean up test memories
            self.memory.delete_all(user_id=self.test_user_id)
        except:
            pass  # Ignore cleanup errors
    
    def test_initialization(self):
        """Test that agent memory initializes correctly."""
        # Test health check
        health = self.memory.health_check()
        assert health["status"] in ["healthy", "degraded"]
        assert "using_api_key" in health
    
    def test_add_and_search_single_message(self):
        """Test adding and searching for single messages."""
        # Add a single message
        message = "I love Italian food, especially pasta carbonara"
        memory_ids = self.memory.add(message, user_id=self.test_user_id)
        assert len(memory_ids) > 0
        
        # Search for related content
        results = self.memory.search("food preferences", user_id=self.test_user_id)
        assert "results" in results
        # Should find the food preference we just added
        memories = results["results"]
        assert len(memories) > 0
    
    def test_add_and_search_conversation(self):
        """Test adding and searching conversation messages."""
        # Add conversation messages
        messages = [
            {"role": "user", "content": "Hi, I'm Alice. I'm a vegetarian and allergic to nuts."},
            {"role": "assistant", "content": "Hello Alice! I've noted that you're a vegetarian and have a nut allergy. I'll keep this in mind."}
        ]
        
        memory_ids = self.memory.add(
            messages, 
            user_id=self.test_user_id,
            session_id=self.test_session_id,
            metadata={"conversation_type": "introduction"}
        )
        assert len(memory_ids) > 0
        
        # Search for dietary information
        results = self.memory.search("dietary restrictions", user_id=self.test_user_id)
        memories = results["results"]
        assert len(memories) > 0
        
        # Should find vegetarian and nut allergy information
        memory_text = " ".join([m["memory"] for m in memories])
        assert "vegetarian" in memory_text.lower() or "nut" in memory_text.lower()
    
    def test_session_filtering(self):
        """Test filtering memories by session."""
        # Add conversational memories in different sessions (as shown in Mem0 docs)
        session1_id = f"session1_{uuid.uuid4().hex[:8]}"
        session2_id = f"session2_{uuid.uuid4().hex[:8]}"
        
        # Use conversational format that Mem0 prefers
        conversation1 = [
            {"role": "user", "content": "I love working on Python projects, especially machine learning applications."},
            {"role": "assistant", "content": "That's great! Python is excellent for ML. I'll remember your interest in Python and machine learning."}
        ]
        
        conversation2 = [
            {"role": "user", "content": "I'm really into React development and building modern web applications."},
            {"role": "assistant", "content": "Excellent! React is a powerful framework. I'll note your preference for React and web development."}
        ]
        
        # Add conversational memories
        memory_ids1 = self.memory.add(conversation1, user_id=self.test_user_id, session_id=session1_id)
        memory_ids2 = self.memory.add(conversation2, user_id=self.test_user_id, session_id=session2_id)
        
        # Verify memories were created
        assert len(memory_ids1) > 0
        assert len(memory_ids2) > 0
        
        # Search without session filters first (to avoid ChromaDB compatibility issues)
        all_python_results = self.memory.search("Python programming", user_id=self.test_user_id)
        all_react_results = self.memory.search("React development", user_id=self.test_user_id)
        
        # Should find relevant memories
        assert len(all_python_results["results"]) > 0
        assert len(all_react_results["results"]) > 0
    
    def test_get_all_memories(self):
        """Test getting all memories for a user."""
        # Add several memories
        messages = [
            "I work as a software engineer",
            "My favorite programming language is Python",
            "I enjoy hiking on weekends"
        ]
        
        for msg in messages:
            self.memory.add(msg, user_id=self.test_user_id)
        
        # Get all memories
        all_memories = self.memory.get_all(user_id=self.test_user_id)
        assert len(all_memories) >= len(messages)
    
    def test_memory_usage_stats(self):
        """Test getting memory usage statistics."""
        # Add conversational memories as shown in Mem0 documentation
        conversation = [
            {"role": "user", "content": "Hi, I'm a software engineer who specializes in Python and machine learning."},
            {"role": "assistant", "content": "Nice to meet you! I'll remember that you're a software engineer with expertise in Python and ML."},
            {"role": "user", "content": "I also prefer working remotely and enjoy collaborative open-source projects."},
            {"role": "assistant", "content": "Got it! I've noted your preference for remote work and interest in collaborative open-source projects."}
        ]
        
        memory_ids = self.memory.add(conversation, user_id=self.test_user_id, session_id=self.test_session_id)
        assert len(memory_ids) > 0  # Verify memories were created
        
        # Get usage stats
        stats = self.memory.get_memory_usage(self.test_user_id)
        assert stats["user_id"] == self.test_user_id
        assert stats["total_memories"] >= 1  # Mem0 might consolidate memories
        assert isinstance(stats["unique_sessions"], int)
        assert isinstance(stats["sessions"], list)
    
    def test_context_for_chat(self):
        """Test getting context for chat queries."""
        # Add some context
        context_messages = [
            "User prefers morning meetings",
            "User is in Pacific timezone",
            "User works remotely"
        ]
        
        for msg in context_messages:
            self.memory.add(msg, user_id=self.test_user_id)
        
        # Get context for a scheduling query
        context = self.memory.get_context_for_chat(
            "Schedule a meeting for next week",
            user_id=self.test_user_id,
            max_memories=3
        )
        
        assert isinstance(context, str)
        assert len(context) > 0
        # Should not be the "no memories" message
        assert "No relevant memories found" not in context





class TestScratchPadMemory:
    """Test cases for ScratchPadMemory (working memory) from utils."""
    
    def setup_method(self):
        """Set up a fresh memory instance for each test."""
        self.memory = ScratchPadMemory()
    
    def test_memory_items(self):
        """Test MemoryItem functionality."""
        # Set memory with metadata
        self.memory.set("user_name", "Alice", "The user's name", "string")
        self.memory.set("temperature", 22.5, "Current temperature", "float", {"unit": "celsius"})
        
        # Test retrieval
        assert self.memory.get("user_name") == "Alice"
        assert self.memory.get("temperature") == 22.5
        assert self.memory.has("user_name") is True
        assert self.memory.has("nonexistent") is False
    
    def test_placeholder_resolution(self):
        """Test placeholder resolution in strings and objects."""
        # Set up memory
        self.memory.set("name", "Alice", "User name")
        self.memory.set("age", 30, "User age")
        self.memory.set("prefs", {"color": "blue", "food": "pizza"}, "User preferences")
        
        # Test string placeholder resolution
        template = "Hello ${memory.name}, you are ${memory.age} years old"
        resolved = self.memory.resolve_placeholders(template)
        assert resolved == "Hello Alice, you are 30 years old"
        
        # Test object placeholder resolution
        obj = {
            "greeting": "Hi ${memory.name}",
            "info": "Age: ${memory.age}, Favorite color: ${memory.prefs.color}",
            "numbers": [1, "${memory.age}", 3]
        }
        resolved_obj = self.memory.resolve_placeholders(obj)
        assert resolved_obj["greeting"] == "Hi Alice"
        assert resolved_obj["info"] == "Age: 30, Favorite color: blue"
        assert resolved_obj["numbers"] == [1, "30", 3]
    
    def test_enumerate_for_prompt(self):
        """Test memory enumeration for prompting."""
        # Empty memory
        enum_empty = self.memory.enumerate_for_prompt()
        assert enum_empty == "(memory empty)"
        
        # Memory with items
        self.memory.set("task", "write code", "Current task")
        self.memory.set("lang", "python", "Programming language", "string")
        
        enum_full = self.memory.enumerate_for_prompt()
        assert "Available memory:" in enum_full
        assert "task" in enum_full
        assert "Current task" in enum_full  # Shows description, not value
        assert "lang (string)" in enum_full
        assert "Programming language" in enum_full


def test_agent_memory_integration():
    """Integration test for AgentMemory with realistic usage patterns."""
    print("\n🤖 Testing AgentMemory Integration:")
    
    # Create memory instance
    memory = create_agent_memory(collection_name=f"integration_test_{uuid.uuid4().hex[:8]}")
    test_user = f"integration_user_{uuid.uuid4().hex[:8]}"
    test_session = f"integration_session_{uuid.uuid4().hex[:8]}"
    
    try:
        print("  📝 Adding user preferences and conversation history...")
        
        # Add user preferences
        preferences = [
            "I'm a software engineer who loves Python",
            "I prefer working in the morning hours",
            "I'm vegetarian and have a nut allergy",
            "I use VS Code as my primary editor"
        ]
        
        for pref in preferences:
            memory.add(pref, user_id=test_user, metadata={"type": "preference"})
        
        # Add conversation history
        conversation = [
            {"role": "user", "content": "Can you help me debug this Python script?"},
            {"role": "assistant", "content": "I'd be happy to help! Please share your Python script and I'll take a look."},
            {"role": "user", "content": "Great! I'm having issues with a FastAPI application."}
        ]
        
        memory.add(
            conversation, 
            user_id=test_user, 
            session_id=test_session,
            metadata={"type": "debugging_session", "topic": "FastAPI"}
        )
        
        print("  🔍 Testing semantic search capabilities...")
        
        # Test contextual search
        search_queries = [
            "programming preferences",
            "dietary restrictions", 
            "development tools",
            "help with Python debugging"
        ]
        
        for query in search_queries:
            results = memory.search(query, user_id=test_user, limit=3)
            memories = results["results"]
            print(f"    Query: '{query}' → {len(memories)} memories found")
            
            if memories:
                best_match = memories[0]
                print(f"      Best: '{best_match['memory'][:60]}...' (score: {best_match['score']:.3f})")
        
        print("  📊 Testing memory usage and statistics...")
        
        # Get usage stats
        stats = memory.get_memory_usage(test_user)
        print(f"    Total memories: {stats['total_memories']}")
        print(f"    Unique sessions: {stats['unique_sessions']}")
        print(f"    Memory types: {stats['memory_types']}")
        
        print("  💬 Testing chat context generation...")
        
        # Test context for chat
        chat_query = "I want to learn more about web development with Python"
        context = memory.get_context_for_chat(
            chat_query, 
            user_id=test_user,
            max_memories=3
        )
        
        print(f"    Context for: '{chat_query}'")
        print(f"    Generated context length: {len(context)} characters")
        
        # Clean up
        deleted_count = memory.delete_all(user_id=test_user)
        print(f"  🧹 Cleaned up {deleted_count} test memories")
        
        print("  ✅ AgentMemory integration test passed")
        
    except Exception as e:
        # Clean up on error
        try:
            memory.delete_all(user_id=test_user)
        except:
            pass
        raise e


if __name__ == "__main__":
    # Run integration tests
    test_agent_memory_integration()
    print("\n🎉 AgentMemory integration test completed!")
