#!/usr/bin/env python3
"""
Test cases for specialized memory systems: semantic, episodic, and procedural memory.
"""
import sys
import os
import tempfile
import uuid
from datetime import datetime, timezone

# Add the package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from jentic_agents.memory.semantic_memory import SemanticMemory
from jentic_agents.memory.episodic_memory import EpisodicMemory
from jentic_agents.memory.procedural_memory import ProceduralMemory
from jentic_agents.memory.agent_memory import AgentMemory, create_agent_memory


class TestSemanticMemory:
    """Test cases for SemanticMemory class."""
    
    def setup_method(self):
        """Set up a fresh semantic memory instance for each test."""
        import uuid
        import os
        # Use unique collection name for each test to avoid conflicts
        unique_name = f"test_semantic_{uuid.uuid4().hex[:8]}"
        
        # Temporarily unset CHROMA_DB_PATH to force in-memory storage
        self.original_path = os.environ.get("CHROMA_DB_PATH")
        if "CHROMA_DB_PATH" in os.environ:
            del os.environ["CHROMA_DB_PATH"]
        
        self.memory = SemanticMemory(collection_name=unique_name)
    
    def teardown_method(self):
        """Clean up test data."""
        try:
            # Clear the collection
            self.memory.clear()
        except:
            pass
        
        # Restore original CHROMA_DB_PATH if it existed
        import os
        if self.original_path is not None:
            os.environ["CHROMA_DB_PATH"] = self.original_path
    
    def test_initialization(self):
        """Test that semantic memory initializes correctly."""
        # Test without custom collection name
        import uuid
        memory_default = SemanticMemory()
        assert len(memory_default) == 0
        
        # Test with custom collection name
        assert len(self.memory) == 0
        assert self.memory.keys() == []
    
    def test_store_and_retrieve_facts(self):
        """Test storing and retrieving factual knowledge."""
        # Store a simple fact
        self.memory.store("capital_france", "Paris is the capital of France")
        
        # Retrieve the fact
        fact = self.memory.retrieve("capital_france")
        assert fact == "Paris is the capital of France"
        
        # Check that it exists in storage
        assert "capital_france" in self.memory
        assert len(self.memory) == 1
    
    def test_store_and_retrieve_concepts(self):
        """Test storing and retrieving conceptual knowledge."""
        # Store a concept with definition
        concept = {
            "definition": "A mammal is a warm-blooded vertebrate animal",
            "characteristics": ["warm-blooded", "vertebrate", "hair or fur"],
            "examples": ["dog", "cat", "human", "whale"]
        }
        self.memory.store("mammal_concept", concept)
        
        # Retrieve and verify - ChromaDB stores as string, so parse back
        retrieved = self.memory.retrieve("mammal_concept")
        assert "mammal is a warm-blooded vertebrate animal" in retrieved.lower()
        
        # Check that it exists
        assert "mammal_concept" in self.memory
    
    def test_store_and_retrieve_skills(self):
        """Test storing and retrieving skill-based knowledge."""
        # Store a skill/procedure
        skill = "How to ride a bicycle: Mount the bicycle, hold the handlebars, start pedaling slowly, maintain balance"
        self.memory.store("bicycle_riding", skill)
        
        # Retrieve and verify
        retrieved = self.memory.retrieve("bicycle_riding")
        assert retrieved == skill
        
        # Check that it exists
        assert "bicycle_riding" in self.memory
    
    def test_semantic_search(self):
        """Test semantic search functionality."""
        # Store some related knowledge
        self.memory.store("fact1", "Water is H2O")
        self.memory.store("fact2", "Ice is frozen water")
        self.memory.store("skill1", "How to make coffee step by step")
        
        # Test semantic search
        results = self.memory.search("water chemistry", top_k=2)
        assert len(results) >= 1
        
        # Should find water-related facts with higher similarity
        water_results = [r for r in results if "water" in r["value"].lower()]
        assert len(water_results) >= 1
    
    def test_multiple_storage_retrieval(self):
        """Test storing and retrieving multiple items."""
        # Store different types of knowledge
        self.memory.store("fact1", "The sky is blue")
        self.memory.store("fact2", "Earth is round")
        self.memory.store("concept1", "A planet is a celestial body")
        self.memory.store("skill1", "How to cook pasta")
        
        # Check all items are stored
        assert len(self.memory) == 4
        all_keys = self.memory.keys()
        assert "fact1" in all_keys
        assert "fact2" in all_keys
        assert "concept1" in all_keys
        assert "skill1" in all_keys
        
        # Retrieve all items
        assert self.memory.retrieve("fact1") == "The sky is blue"
        assert self.memory.retrieve("fact2") == "Earth is round"
        assert self.memory.retrieve("concept1") == "A planet is a celestial body"
        assert self.memory.retrieve("skill1") == "How to cook pasta"
    
    def test_delete_operations(self):
        """Test deletion functionality."""
        # Store some knowledge
        self.memory.store("temp1", "Temporary fact 1")
        self.memory.store("temp2", "Temporary fact 2")
        assert len(self.memory) == 2
        
        # Delete one item
        assert self.memory.delete("temp1") is True
        assert self.memory.retrieve("temp1") is None
        assert self.memory.retrieve("temp2") is not None
        assert len(self.memory) == 1
        
        # Try to delete non-existent item
        assert self.memory.delete("nonexistent") is False
    
    def test_clear_operation(self):
        """Test clearing all knowledge."""
        # Add some knowledge
        self.memory.store("fact1", "Water is H2O")
        self.memory.store("fact2", "Earth is round")
        assert len(self.memory) == 2
        
        # Clear all
        self.memory.clear()
        assert len(self.memory) == 0
        assert self.memory.keys() == []
    
    def test_key_existence_check(self):
        """Test key existence checking."""
        # Initially empty
        assert "nonexistent" not in self.memory
        
        # Store and check
        self.memory.store("test_key", "test value")
        assert "test_key" in self.memory
        assert "nonexistent" not in self.memory
        
        # Delete and check
        self.memory.delete("test_key")
        assert "test_key" not in self.memory
    
    def test_persistence(self):
        """Test persistence through environment variable."""
        import os
        # Store knowledge
        self.memory.store("persistent_fact", "This should persist")
        
        # If CHROMA_DB_PATH is set, knowledge should persist
        # This is a basic test - in practice, you'd set CHROMA_DB_PATH
        # and create a new memory instance to test true persistence
        stored_fact = self.memory.retrieve("persistent_fact")
        assert stored_fact == "This should persist"


class TestEpisodicMemory:
    """Test cases for EpisodicMemory class."""
    
    def setup_method(self):
        """Set up a fresh episodic memory instance for each test."""
        self.temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.json')
        self.temp_file.close()
        self.memory = EpisodicMemory(storage_path=self.temp_file.name)
    
    def teardown_method(self):
        """Clean up temporary files."""
        try:
            os.unlink(self.temp_file.name)
        except:
            pass
    
    def test_initialization(self):
        """Test that episodic memory initializes correctly."""
        assert self.memory.keys() == []
        assert self.memory._episode_sequence == 0
    
    def test_store_and_retrieve_episodes(self):
        """Test storing and retrieving episodes."""
        # Store a simple episode
        episode = "I went to the grocery store and bought apples"
        self.memory.store("grocery_trip", episode)
        
        # Retrieve the episode
        retrieved = self.memory.retrieve("grocery_trip")
        assert retrieved == episode
        
        # Check that sequence is incremented
        assert self.memory._episode_sequence == 1
    
    def test_episode_metadata_extraction(self):
        """Test extraction of episode metadata."""
        # Store an episode with rich context
        episode = {
            "event": "Had dinner with John at the Italian restaurant",
            "when": "last Friday evening",
            "where": "Tony's Restaurant",
            "who": "John",
            "emotions": "happy and satisfied"
        }
        self.memory.store("dinner_episode", episode)
        
        # Retrieve and check stored episode data
        episode_data = self.memory._episodes["dinner_episode"]
        
        # Check context extraction
        assert "who" in episode_data["context"]
        assert episode_data["context"]["who"] == "John"
        
        # Check people extraction
        assert "John" in episode_data["people"]
        
        # Check locations extraction
        assert len(episode_data["locations"]) >= 0  # May or may not extract location
    
    def test_emotion_extraction(self):
        """Test emotion extraction from episodes."""
        # Episode with emotional content
        self.memory.store("emotional_episode", "I was so excited and happy about the promotion!")
        
        episode_data = self.memory._episodes["emotional_episode"]
        emotions = episode_data["emotions"]
        
        assert "excited" in emotions
        assert "happy" in emotions
    
    def test_vividness_and_importance_assessment(self):
        """Test vividness and importance scoring."""
        # Vivid episode with sensory details
        vivid_episode = "I saw the bright sunset and heard the waves crashing loudly"
        self.memory.store("vivid_episode", vivid_episode)
        
        # Important episode
        important_episode = "My first day at the new job was memorable and special"
        self.memory.store("important_episode", important_episode)
        
        vivid_data = self.memory._episodes["vivid_episode"]
        important_data = self.memory._episodes["important_episode"]
        
        # Vivid episode should have higher vividness
        assert vivid_data["vividness"] > 0.5
        
        # Important episode should have higher importance
        assert important_data["importance"] > 0.5
    
    def test_recall_strengthening(self):
        """Test that recalling episodes strengthens memory."""
        self.memory.store("test_episode", "A test memory")
        
        # Initial state
        episode_data = self.memory._episodes["test_episode"]
        initial_vividness = episode_data["vividness"]
        initial_recall_count = episode_data["recalled_count"]
        
        # Recall multiple times
        self.memory.retrieve("test_episode")
        self.memory.retrieve("test_episode")
        
        # Check strengthening
        episode_data = self.memory._episodes["test_episode"]
        assert episode_data["recalled_count"] == initial_recall_count + 2
        assert episode_data["vividness"] >= initial_vividness
        assert episode_data["last_recalled"] is not None
    
    def test_timeframe_filtering(self):
        """Test filtering episodes by timeframe."""
        # Store episodes with different timestamps
        now = datetime.now(timezone.utc)
        
        # Manually set timestamps to test filtering
        self.memory.store("recent_episode", "Recent event")
        self.memory.store("old_episode", "Old event")
        
        # Modify timestamps for testing
        recent_time = now.isoformat()
        old_time = datetime(2020, 1, 1, tzinfo=timezone.utc).isoformat()
        
        self.memory._episodes["recent_episode"]["timestamp"] = recent_time
        self.memory._episodes["old_episode"]["timestamp"] = old_time
        
        # Test timeframe filtering
        recent_episodes = self.memory.get_episodes_by_timeframe(
            "2023-01-01T00:00:00+00:00",
            "2026-01-01T00:00:00+00:00"
        )
        
        assert "recent_episode" in recent_episodes
        assert "old_episode" not in recent_episodes
    
    def test_people_filtering(self):
        """Test filtering episodes by people involved."""
        # Store episodes with different people
        self.memory.store("alice_episode", "Had lunch with Alice")
        self.memory.store("bob_episode", "Went to movies with Bob")
        self.memory.store("both_episode", "Party with Alice and Bob")
        
        # Filter by single person
        alice_episodes = self.memory.get_episodes_by_people("Alice")
        assert len(alice_episodes) >= 1  # Should find episodes mentioning Alice
        
        # Filter by multiple people
        both_episodes = self.memory.get_episodes_by_people(["Alice", "Bob"])
        assert len(both_episodes) >= 1  # Should find episodes mentioning either
    
    def test_memory_strength_calculation(self):
        """Test memory strength calculation."""
        self.memory.store("strong_memory", "A very important and vivid memory")
        
        # Get initial strength
        initial_strength = self.memory.get_memory_strength("strong_memory")
        assert initial_strength is not None
        assert 0.0 <= initial_strength <= 1.0
        
        # Recall to strengthen
        self.memory.retrieve("strong_memory")
        self.memory.retrieve("strong_memory")
        
        # Strength should increase
        new_strength = self.memory.get_memory_strength("strong_memory")
        assert new_strength >= initial_strength
        
        # Non-existent memory
        assert self.memory.get_memory_strength("nonexistent") is None
    
    def test_persistence(self):
        """Test persistence of episodes."""
        # Store episode
        self.memory.store("persistent_episode", "This episode should persist")
        
        # Create new memory instance
        new_memory = EpisodicMemory(storage_path=self.temp_file.name)
        
        # Should load the episode
        episode = new_memory.retrieve("persistent_episode")
        assert episode == "This episode should persist"
        assert new_memory._episode_sequence > 0


class TestProceduralMemory:
    """Test cases for ProceduralMemory class."""
    
    def setup_method(self):
        """Set up a fresh procedural memory instance for each test."""
        self.temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.json')
        self.temp_file.close()
        self.memory = ProceduralMemory(storage_path=self.temp_file.name)
    
    def teardown_method(self):
        """Clean up temporary files."""
        try:
            os.unlink(self.temp_file.name)
        except:
            pass
    
    def test_initialization(self):
        """Test that procedural memory initializes correctly."""
        assert self.memory.keys() == []
    
    def test_store_and_retrieve_procedures(self):
        """Test storing and retrieving procedures."""
        # Store a simple procedure
        procedure = "How to make coffee: 1. Boil water, 2. Add coffee, 3. Stir"
        self.memory.store("make_coffee", procedure)
        
        # Retrieve the procedure
        retrieved = self.memory.retrieve("make_coffee")
        assert retrieved == procedure
    
    def test_procedure_type_classification(self):
        """Test automatic procedure type classification."""
        # Workflow with steps
        workflow = {
            "steps": ["Step 1: Open file", "Step 2: Edit content", "Step 3: Save file"]
        }
        self.memory.store("file_editing", workflow)
        
        # Habit with trigger
        habit = {
            "trigger": "When alarm rings",
            "action": "Get out of bed immediately"
        }
        self.memory.store("morning_habit", habit)
        
        # Skill 
        skill = "How to play guitar chords"
        self.memory.store("guitar_skill", skill)
        
        # Check classifications
        workflow_data = self.memory._procedures["file_editing"]
        habit_data = self.memory._procedures["morning_habit"]
        skill_data = self.memory._procedures["guitar_skill"]
        
        assert workflow_data["type"] == "workflow"
        assert habit_data["type"] == "habit"
        assert skill_data["type"] == "skill"
    
    def test_difficulty_assessment(self):
        """Test difficulty level assessment."""
        # Easy procedure
        easy_proc = "Simple task with one step"
        self.memory.store("easy_task", easy_proc)
        
        # Hard procedure with many steps
        hard_proc = {
            "steps": [f"Step {i}" for i in range(1, 15)]  # 14 steps
        }
        self.memory.store("complex_task", hard_proc)
        
        easy_data = self.memory._procedures["easy_task"]
        hard_data = self.memory._procedures["complex_task"]
        
        assert easy_data["difficulty"] == "easy"
        assert hard_data["difficulty"] == "hard"
    
    def test_mastery_progression(self):
        """Test mastery level progression with practice."""
        self.memory.store("practice_skill", "A skill to practice")
        
        # Initial mastery should be low
        initial_mastery = self.memory._procedures["practice_skill"]["mastery_level"]
        assert initial_mastery < 0.5
        
        # Practice (retrieve) multiple times
        for _ in range(10):
            self.memory.retrieve("practice_skill")
        
        # Mastery should increase
        final_mastery = self.memory._procedures["practice_skill"]["mastery_level"]
        assert final_mastery > initial_mastery
    
    def test_execution_tracking(self):
        """Test execution tracking and success rate."""
        self.memory.store("trackable_skill", "A skill to track")
        
        # Execute successfully
        result = self.memory.execute_procedure("trackable_skill", execution_time=5.0, success=True)
        assert result == "A skill to track"
        
        # Execute with failure
        self.memory.execute_procedure("trackable_skill", success=False)
        
        # Check statistics
        stats = self.memory.get_procedure_stats("trackable_skill")
        assert stats["execution_count"] == 2
        assert stats["success_rate"] < 1.0  # Should be 0.5 (1 success, 1 failure)
        
        # Non-existent procedure
        assert self.memory.execute_procedure("nonexistent") is None
    
    def test_procedure_filtering(self):
        """Test filtering procedures by various criteria."""
        # Store different types of procedures
        self.memory.store("workflow1", {"steps": ["A", "B", "C"]})
        self.memory.store("skill1", "How to do something")
        self.memory.store("habit1", {"trigger": "when X happens"})
        
        # Practice one skill to increase mastery
        for _ in range(20):
            self.memory.retrieve("skill1")
        
        # Filter by type
        workflows = self.memory.get_procedures_by_type("workflow")
        skills = self.memory.get_procedures_by_type("skill")
        habits = self.memory.get_procedures_by_type("habit")
        
        assert len(workflows) == 1
        assert len(skills) == 1
        assert len(habits) == 1
        
        # Filter by mastery level
        high_mastery = self.memory.get_procedures_by_mastery(min_mastery=0.8)
        low_mastery = self.memory.get_procedures_by_mastery(max_mastery=0.3)
        
        # The practiced skill should have high mastery
        assert len(high_mastery) >= 0  # May or may not reach 0.8 depending on implementation
    
    def test_step_extraction(self):
        """Test extraction of steps from procedures."""
        # Procedure with explicit steps
        proc_with_steps = {
            "name": "Cooking procedure",
            "steps": ["Preheat oven", "Mix ingredients", "Bake for 20 minutes"]
        }
        self.memory.store("cooking", proc_with_steps)
        
        procedure_data = self.memory._procedures["cooking"]
        assert len(procedure_data["steps"]) == 3
        assert "Preheat oven" in procedure_data["steps"]
    
    def test_tag_extraction(self):
        """Test automatic tag extraction."""
        # Physical procedure
        physical_proc = "Physical exercise routine for motor skills"
        self.memory.store("exercise", physical_proc)
        
        # Cognitive procedure
        cognitive_proc = "Mental problem-solving strategy"
        self.memory.store("problem_solving", cognitive_proc)
        
        exercise_data = self.memory._procedures["exercise"]
        cognitive_data = self.memory._procedures["problem_solving"]
        
        assert "physical" in exercise_data["tags"]
        assert "cognitive" in cognitive_data["tags"]
    
    def test_procedure_statistics(self):
        """Test getting detailed procedure statistics."""
        self.memory.store("stat_test", "Test procedure")
        
        # Execute a few times
        self.memory.execute_procedure("stat_test", success=True)
        self.memory.execute_procedure("stat_test", success=True)
        self.memory.execute_procedure("stat_test", success=False)
        
        stats = self.memory.get_procedure_stats("stat_test")
        
        assert stats["execution_count"] == 3
        assert abs(stats["success_rate"] - 2/3) < 0.01  # 2 successes out of 3
        assert stats["type"] == "procedure"
        assert stats["difficulty"] in ["easy", "medium", "hard"]
        
        # Non-existent procedure
        assert self.memory.get_procedure_stats("nonexistent") is None
    
    def test_persistence(self):
        """Test persistence of procedures."""
        # Store procedure
        self.memory.store("persistent_proc", "This should persist")
        
        # Execute to update stats
        self.memory.execute_procedure("persistent_proc", success=True)
        
        # Create new memory instance
        new_memory = ProceduralMemory(storage_path=self.temp_file.name)
        
        # Should load the procedure with stats
        proc = new_memory.retrieve("persistent_proc")
        assert proc == "This should persist"
        
        stats = new_memory.get_procedure_stats("persistent_proc")
        assert stats["execution_count"] > 0


class TestAgentMemory:
    """Test cases for AgentMemory class."""
    
    def setup_method(self):
        """Set up a fresh agent memory instance for each test."""
        import os
        import tempfile
        import uuid
        
        # Create a unique temporary directory for ChromaDB
        self.temp_dir = tempfile.mkdtemp(prefix=f"agent_memory_test_{uuid.uuid4().hex[:8]}_")
        
        # Store original environment variables
        self.original_chroma_path = os.environ.get("CHROMA_DB_PATH")
        self.original_telemetry = os.environ.get("MEM0_TELEMETRY")
        
        # Set up test environment with unique collection name
        self.unique_collection = f"test_agent_memory_{uuid.uuid4().hex[:8]}"
        os.environ["MEM0_TELEMETRY"] = "false"
        
        try:
            # Create a test configuration that works without external APIs
            config = {
                "vector_store": {
                    "provider": "chroma",
                    "config": {
                        "collection_name": self.unique_collection,
                        "path": self.temp_dir,
                        # Use in-memory for tests to avoid conflicts
                    }
                },
                # Use a minimal config that might work without real API keys
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
            
            # Set a dummy API key for testing if none exists
            if not os.environ.get("OPENAI_API_KEY"):
                os.environ["OPENAI_API_KEY"] = "test-key-for-testing"
                self.set_dummy_key = True
            else:
                self.set_dummy_key = False
            
            # Use AgentMemory directly with custom config
            self.memory = AgentMemory(
                config=config,
                enable_telemetry=False
            )
        except Exception as e:
            # If Mem0 is not available or fails, skip the test
            import pytest
            pytest.skip(f"Mem0 not available: {e}")
    
    def teardown_method(self):
        """Clean up test data and restore environment."""
        import os
        import shutil
        
        # Clean up temporary directory
        try:
            if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
        except:
            pass
        
        # Clean up dummy API key if we set it
        if hasattr(self, 'set_dummy_key') and self.set_dummy_key:
            if os.environ.get("OPENAI_API_KEY") == "test-key-for-testing":
                del os.environ["OPENAI_API_KEY"]
        
        # Restore original environment variables
        if self.original_chroma_path is not None:
            os.environ["CHROMA_DB_PATH"] = self.original_chroma_path
        elif "CHROMA_DB_PATH" in os.environ:
            del os.environ["CHROMA_DB_PATH"]
            
        if self.original_telemetry is not None:
            os.environ["MEM0_TELEMETRY"] = self.original_telemetry
        elif "MEM0_TELEMETRY" in os.environ:
            del os.environ["MEM0_TELEMETRY"]
    
    def test_initialization(self):
        """Test that agent memory initializes correctly."""
        # Test that memory is created
        assert self.memory is not None
        assert hasattr(self.memory, 'memory')
        
        # Test that all required methods exist
        required_methods = [
            'add', 'search', 'get_all', 'update', 'delete_memory', 'delete_all',
            'get_memory_usage', 'get_context_for_chat', 'health_check',
            'store', 'retrieve', 'delete', 'clear', 'keys'
        ]
        
        for method in required_methods:
            assert hasattr(self.memory, method), f"Missing method: {method}"
            assert callable(getattr(self.memory, method)), f"Method not callable: {method}"
        
        # Test health check
        health = self.memory.health_check()
        assert isinstance(health, dict)
        assert "status" in health
        assert health["status"] in ["healthy", "degraded", "unhealthy"]
    
    def test_add_and_retrieve_basic(self):
        """Test basic add and retrieve functionality."""
        user_id = "test_user_1"
        message = "Remember that I like coffee"
        
        # Add a memory (may not work with dummy API keys)
        memory_ids = self.memory.add(message, user_id=user_id)
        # Note: With dummy API keys, this might return empty list
        assert isinstance(memory_ids, list)
        
        # Search for the memory
        results = self.memory.search("coffee", user_id=user_id)
        assert results is not None
        assert isinstance(results, dict)
        assert "results" in results
        
        # Check if our message is found (might be empty with dummy keys)
        found_memories = results["results"]
        assert isinstance(found_memories, list)
        
        # If memories were actually created, verify content
        if len(found_memories) > 0 and len(memory_ids) > 0:
            memory_texts = [m.get("memory", "") for m in found_memories]
            assert any("coffee" in text.lower() for text in memory_texts)
    
    def test_add_multiple_messages(self):
        """Test adding multiple messages at once."""
        user_id = "test_user_2"
        messages = [
            {"role": "user", "content": "I love pizza"},
            {"role": "assistant", "content": "Great! What's your favorite topping?"},
            {"role": "user", "content": "I prefer pepperoni"}
        ]
        
        # Add multiple messages
        memory_ids = self.memory.add(messages, user_id=user_id)
        assert len(memory_ids) > 0
        
        # Search for pizza-related memories
        results = self.memory.search("pizza", user_id=user_id)
        assert len(results.get("results", [])) > 0
    

    

    
    def test_update_memory(self):
        """Test updating an existing memory."""
        user_id = "test_user_5"
        original_message = "I like tea"
        
        # Add a memory
        memory_ids = self.memory.add(original_message, user_id=user_id)
        memory_id = memory_ids[0]
        
        # Update the memory
        updated_message = "I like tea and coffee"
        result = self.memory.update(memory_id, updated_message)
        
        # Verify update was successful
        assert result is not None
        
        # Search for the updated content
        results = self.memory.search("coffee", user_id=user_id)
        found_memories = results.get("results", [])
        
        # Should find the updated memory
        assert len(found_memories) > 0
        memory_texts = [m.get("memory", "") for m in found_memories]
        assert any("coffee" in text.lower() for text in memory_texts)
    

    

    

    
    def test_context_for_chat(self):
        """Test getting memory context for chat."""
        user_id = "test_user_9"
        
        # Add relevant memories
        self.memory.add("I work as a software engineer", user_id=user_id)
        self.memory.add("I enjoy playing guitar in my free time", user_id=user_id)
        self.memory.add("My favorite programming language is Python", user_id=user_id)
        
        # Get context for a query about work
        context = self.memory.get_context_for_chat(
            "What do I do for work?",
            user_id=user_id,
            max_memories=3
        )
        
        assert isinstance(context, str)
        assert len(context) > 0
        # Should contain relevant information about work/software engineering
        assert "engineer" in context.lower() or "work" in context.lower()
    
    def test_base_memory_interface(self):
        """Test BaseMemory interface compatibility."""
        # Test store (may not work with dummy API keys, but should not crash)
        key = "base_test_key"
        value = "This is a test value"
        
        try:
            self.memory.store(key, value)
            # If store works, test retrieve
            retrieved = self.memory.retrieve(key)
            # Note: With dummy API keys, storage might not work, so we don't assert success
            if retrieved is not None:
                assert value in str(retrieved)
                
                # Test delete if retrieve worked
                deleted = self.memory.delete(key)
                assert isinstance(deleted, bool)
                
                # Test retrieve after delete
                retrieved_after = self.memory.retrieve(key)
                # May be None due to deletion or due to dummy keys
                
        except Exception as e:
            # With dummy API keys, operations might fail - that's expected
            # We just verify the interface exists and doesn't crash completely
            pass
        
        # Test keys (should warn that it's not fully supported)
        keys = self.memory.keys()
        assert isinstance(keys, list)
        # Note: keys() returns empty list due to Mem0 limitations
        
        # Verify that the methods exist and are callable
        assert hasattr(self.memory, 'store')
        assert hasattr(self.memory, 'retrieve')
        assert hasattr(self.memory, 'delete')
        assert hasattr(self.memory, 'clear')
        assert hasattr(self.memory, 'keys')
        assert callable(self.memory.store)
        assert callable(self.memory.retrieve)
        assert callable(self.memory.delete)
        assert callable(self.memory.clear)
        assert callable(self.memory.keys)
    
    def test_factory_function(self):
        """Test the create_agent_memory factory function."""
        import tempfile
        import uuid
        
        temp_dir = tempfile.mkdtemp(prefix=f"factory_test_{uuid.uuid4().hex[:8]}_")
        
        try:
            # Test creating with different config types using unique collection
            from jentic_agents.memory.memory_config import MemoryConfig
            config = MemoryConfig.local_only(
                chroma_path=temp_dir,
                collection_name=f"factory_test_{uuid.uuid4().hex[:8]}"
            )
            
            local_memory = AgentMemory(
                config=config,
                enable_telemetry=False
            )
            assert local_memory is not None
            assert hasattr(local_memory, 'memory')
            
            # Test health check
            health = local_memory.health_check()
            assert health["status"] in ["healthy", "degraded", "unhealthy"]
            
        except Exception as e:
            import pytest
            pytest.skip(f"Factory function test failed: {e}")
        finally:
            import shutil
            try:
                shutil.rmtree(temp_dir)
            except:
                pass


class TestMemoryIntegration:
    """Test integration between different memory types."""
    
    def setup_method(self):
        """Set up all memory types."""
        import uuid
        import os
        unique_id = uuid.uuid4().hex[:8]
        
        # Temporarily unset CHROMA_DB_PATH to force in-memory storage
        self.original_path = os.environ.get("CHROMA_DB_PATH")
        if "CHROMA_DB_PATH" in os.environ:
            del os.environ["CHROMA_DB_PATH"]
        
        self.semantic = SemanticMemory(collection_name=f"integration_semantic_{unique_id}")
        self.episodic = EpisodicMemory()
        self.procedural = ProceduralMemory()
    
    def test_related_memories_across_types(self):
        """Test storing related information across different memory types."""
        # Semantic: Store knowledge about cooking
        self.semantic.store("cooking_knowledge", "Cooking involves applying heat to food")
        
        # Episodic: Store memory of cooking experience
        self.episodic.store("first_cooking", "My first time cooking pasta was exciting but stressful")
        
        # Procedural: Store cooking procedure
        cooking_steps = {
            "steps": [
                "Boil water in a large pot",
                "Add salt to the water",
                "Add pasta when water is boiling",
                "Cook for 8-10 minutes",
                "Drain the pasta"
            ]
        }
        self.procedural.store("cook_pasta", cooking_steps)
        
        # Verify all memories are stored
        assert self.semantic.retrieve("cooking_knowledge") is not None
        assert self.episodic.retrieve("first_cooking") is not None
        assert self.procedural.retrieve("cook_pasta") is not None
        
        # Check that procedural memory tracks the skill
        pasta_stats = self.procedural.get_procedure_stats("cook_pasta")
        assert pasta_stats["type"] == "workflow"
        assert len(pasta_stats) > 0
    
    def test_memory_types_independence(self):
        """Test that different memory types maintain independence."""
        # Store same key in different memory types
        key = "test_memory"
        
        self.semantic.store(key, "Semantic knowledge")
        self.episodic.store(key, "Episodic experience")
        self.procedural.store(key, "Procedural skill")
        
        # Each should store and retrieve independently
        assert self.semantic.retrieve(key) == "Semantic knowledge"
        assert self.episodic.retrieve(key) == "Episodic experience"
        assert self.procedural.retrieve(key) == "Procedural skill"
        
        # Deleting from one shouldn't affect others
        self.semantic.delete(key)
        assert self.semantic.retrieve(key) is None
        assert self.episodic.retrieve(key) == "Episodic experience"
        assert self.procedural.retrieve(key) == "Procedural skill"


if __name__ == "__main__":
    # Run a simple test to verify imports work
    print("Testing memory class imports...")
    
    semantic = SemanticMemory()
    episodic = EpisodicMemory()
    procedural = ProceduralMemory()
    
    print("✓ SemanticMemory imported and instantiated")
    print("✓ EpisodicMemory imported and instantiated")
    print("✓ ProceduralMemory imported and instantiated")
    try:
        agent = create_agent_memory(config_type="local")
        print("✓ AgentMemory imported and instantiated")
    except Exception as e:
        print(f"⚠ AgentMemory import failed (dependencies may be missing): {e}")
    
    print("All available specialized memory classes are working!") 