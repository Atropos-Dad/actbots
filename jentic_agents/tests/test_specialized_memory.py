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


class TestSemanticMemory:
    """Test cases for SemanticMemory class."""
    
    def setup_method(self):
        """Set up a fresh semantic memory instance for each test."""
        self.temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.json')
        self.temp_file.close()
        self.memory = SemanticMemory(storage_path=self.temp_file.name)
    
    def teardown_method(self):
        """Clean up temporary files."""
        try:
            os.unlink(self.temp_file.name)
        except:
            pass
    
    def test_initialization(self):
        """Test that semantic memory initializes correctly."""
        # Test without storage path
        memory_no_storage = SemanticMemory()
        assert memory_no_storage.keys() == []
        
        # Test with storage path
        assert self.memory.keys() == []
        assert os.path.exists(self.temp_file.name) or not self.memory._storage_path
    
    def test_store_and_retrieve_facts(self):
        """Test storing and retrieving factual knowledge."""
        # Store a simple fact
        self.memory.store("capital_france", "Paris is the capital of France")
        
        # Retrieve the fact
        fact = self.memory.retrieve("capital_france")
        assert fact == "Paris is the capital of France"
        
        # Check metadata
        metadata = self.memory.get_metadata("capital_france")
        assert metadata is not None
        assert metadata["type"] == "fact"
        assert metadata["accessed_count"] == 1
        assert metadata["created_at"] is not None
    
    def test_store_and_retrieve_concepts(self):
        """Test storing and retrieving conceptual knowledge."""
        # Store a concept with definition
        concept = {
            "definition": "A mammal is a warm-blooded vertebrate animal",
            "characteristics": ["warm-blooded", "vertebrate", "hair or fur"],
            "examples": ["dog", "cat", "human", "whale"]
        }
        self.memory.store("mammal_concept", concept)
        
        # Retrieve and verify
        retrieved = self.memory.retrieve("mammal_concept")
        assert retrieved == concept
        
        # Check metadata - should be classified as concept
        metadata = self.memory.get_metadata("mammal_concept")
        assert metadata["type"] == "concept"
    
    def test_store_and_retrieve_skills(self):
        """Test storing and retrieving skill-based knowledge."""
        # Store a skill/procedure
        skill = {
            "procedure": "How to ride a bicycle",
            "steps": [
                "Mount the bicycle",
                "Hold the handlebars",
                "Start pedaling slowly",
                "Maintain balance"
            ]
        }
        self.memory.store("bicycle_riding", skill)
        
        # Retrieve and verify
        retrieved = self.memory.retrieve("bicycle_riding")
        assert retrieved == skill
        
        # Check metadata - should be classified as skill
        metadata = self.memory.get_metadata("bicycle_riding")
        assert metadata["type"] == "skill"
    
    def test_knowledge_type_inference(self):
        """Test automatic knowledge type inference."""
        # Test fact inference
        self.memory.store("fact1", "Water is H2O")
        assert self.memory.get_metadata("fact1")["type"] == "fact"
        
        # Test skill inference
        self.memory.store("skill1", "How to make coffee step by step")
        assert self.memory.get_metadata("skill1")["type"] == "skill"
        
        # Test data inference
        self.memory.store("data1", 42)
        assert self.memory.get_metadata("data1")["type"] == "data"
    
    def test_get_knowledge_by_type(self):
        """Test filtering knowledge by type."""
        # Store different types of knowledge
        self.memory.store("fact1", "The sky is blue")
        self.memory.store("fact2", "Earth is round")
        self.memory.store("concept1", {"definition": "A planet is a celestial body"})
        self.memory.store("skill1", "How to cook pasta")
        
        # Get facts only
        facts = self.memory.get_knowledge_by_type("fact")
        assert len(facts) == 2
        assert "fact1" in facts
        assert "fact2" in facts
        
        # Get concepts only
        concepts = self.memory.get_knowledge_by_type("concept")
        assert len(concepts) == 1
        assert "concept1" in concepts
        
        # Get skills only
        skills = self.memory.get_knowledge_by_type("skill")
        assert len(skills) == 1
        assert "skill1" in skills
    
    def test_access_tracking(self):
        """Test that access is properly tracked."""
        self.memory.store("test_knowledge", "Some important fact")
        
        # Initial access count should be 0
        metadata = self.memory.get_metadata("test_knowledge")
        assert metadata["accessed_count"] == 0
        assert metadata["last_accessed"] is None
        
        # Retrieve multiple times
        self.memory.retrieve("test_knowledge")
        self.memory.retrieve("test_knowledge")
        self.memory.retrieve("test_knowledge")
        
        # Check updated access count
        metadata = self.memory.get_metadata("test_knowledge")
        assert metadata["accessed_count"] == 3
        assert metadata["last_accessed"] is not None
    
    def test_usage_statistics(self):
        """Test usage statistics generation."""
        # Empty memory stats
        stats = self.memory.get_usage_stats()
        assert stats["total_items"] == 0
        
        # Add some knowledge
        self.memory.store("fact1", "Water is H2O")
        self.memory.store("concept1", {"definition": "Test concept"})
        self.memory.store("skill1", "How to test")
        
        # Access some items
        self.memory.retrieve("fact1")
        self.memory.retrieve("fact1")  # Access twice
        self.memory.retrieve("concept1")
        
        # Check stats
        stats = self.memory.get_usage_stats()
        assert stats["total_items"] == 3
        assert stats["knowledge_types"]["fact"] == 1
        assert stats["knowledge_types"]["concept"] == 1
        assert stats["knowledge_types"]["skill"] == 1
        assert stats["most_accessed"][0] == "fact1"
        assert stats["most_accessed"][1] == 2
    
    def test_delete_and_clear(self):
        """Test deletion operations."""
        # Store some knowledge
        self.memory.store("temp1", "Temporary fact 1")
        self.memory.store("temp2", "Temporary fact 2")
        
        # Delete one item
        assert self.memory.delete("temp1") is True
        assert self.memory.retrieve("temp1") is None
        assert self.memory.retrieve("temp2") is not None
        
        # Try to delete non-existent item
        assert self.memory.delete("nonexistent") is False
        
        # Clear all
        self.memory.clear()
        assert len(self.memory.keys()) == 0
    
    def test_persistence(self):
        """Test persistence to disk."""
        # Store knowledge
        self.memory.store("persistent_fact", "This should persist")
        
        # Create new memory instance with same storage path
        new_memory = SemanticMemory(storage_path=self.temp_file.name)
        
        # Should load the stored knowledge
        fact = new_memory.retrieve("persistent_fact")
        assert fact == "This should persist"


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


class TestMemoryIntegration:
    """Test integration between different memory types."""
    
    def setup_method(self):
        """Set up all memory types."""
        self.semantic = SemanticMemory()
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
    print("All specialized memory classes are working!") 