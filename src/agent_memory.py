import sqlite3
from utils import Utils
from openai import OpenAI
from prompts import AgentPrompts
import logging
import time

class AgentMemory:
    """Handles memory and reflection functionality for agent users."""
    
    MEMORY_TYPE_INTERACTION = 'interaction'
    MEMORY_TYPE_REFLECTION = 'reflection'
    VALID_MEMORY_TYPES = {MEMORY_TYPE_INTERACTION, MEMORY_TYPE_REFLECTION}
    
    def __init__(self, user_id: str, conn: sqlite3.Connection, persona: dict, memory_decay_rate: float = 0.1):
        self.user_id = user_id
        self.conn = conn
        self.cursor = conn.cursor()
        self.persona = persona
        self.memory_decay_rate = memory_decay_rate
        self.memory_importance_threshold = 0.3
    
    def add_memory(self, content: str, memory_type: str = MEMORY_TYPE_INTERACTION, importance_score: float = 0.0) -> str:
        """Add a new memory for the agent."""
        if memory_type not in self.VALID_MEMORY_TYPES:
            raise ValueError(f"Invalid memory_type. Must be one of: {', '.join(self.VALID_MEMORY_TYPES)}")
        
        memory_id = Utils.generate_formatted_id("memory", self.conn)
        
        if importance_score == 0.0:
            importance_score = self._evaluate_memory_importance(content)
        
        self.cursor.execute('''
            INSERT INTO agent_memories (
                memory_id, user_id, memory_type, content, 
                importance_score, created_at
            )
            VALUES (?, ?, ?, ?, ?, datetime('now'))
        ''', (memory_id, self.user_id, memory_type, content, importance_score))
        
        self.conn.commit()
        return memory_id
    
    def get_relevant_memories(self, memory_type: str = MEMORY_TYPE_INTERACTION, limit: int = 5) -> list[dict]:
        """Retrieve relevant memories, optionally filtered by memory type."""
        if memory_type not in self.VALID_MEMORY_TYPES:
            raise ValueError(f"Invalid memory_type. Must be one of: {', '.join(self.VALID_MEMORY_TYPES)}")
        
        self._decay_memories()
        
        query = '''
            SELECT memory_id, content, memory_type, importance_score, created_at, decay_factor
            FROM agent_memories
            WHERE user_id = ? 
            AND importance_score * decay_factor >= ?
            AND memory_type = ?
            ORDER BY importance_score * decay_factor DESC 
            LIMIT ?
        '''
        
        self.cursor.execute(query, (
            self.user_id, 
            self.memory_importance_threshold,
            memory_type,
            limit
        ))
        
        return [{
            'id': row[0],
            'content': row[1],
            'type': row[2],
            'importance': row[3],
            'created_at': row[4],
            'decay_factor': row[5]
        } for row in self.cursor.fetchall()]
    
    def reflect(self, openai_client: OpenAI, engine: str, temperature: float):
        """Generate reflections based on recent memories and experiences."""
        recent_memories = self.get_relevant_memories(memory_type=self.MEMORY_TYPE_INTERACTION, limit=10)
        
        if not recent_memories:
            return
        
        memory_text = "\n".join([
            f"- {mem['content']} ({mem['type']})"
            for mem in recent_memories
        ])
        
        prompt = AgentPrompts.create_reflection_prompt(self.persona, memory_text)
        reflection = Utils.generate_llm_response(
            openai_client,
            engine,
            prompt,
            "You are helping an AI agent reflect on its recent experiences and form insights.",
            temperature=temperature,
            max_tokens=200
        )
        
        self.add_memory(reflection, memory_type=self.MEMORY_TYPE_REFLECTION, importance_score=0.8)
    
    def _decay_memories(self):
        """Apply decay to all memories based on time passed."""
        try:
            self.cursor.execute('''
                UPDATE agent_memories
                SET decay_factor = MAX(0, decay_factor - ? * 
                    (julianday('now') - julianday(last_accessed))),
                    last_accessed = datetime('now')  -- Update last_accessed time
                WHERE user_id = ?
            ''', (self.memory_decay_rate, self.user_id))
        except sqlite3.OperationalError as e:
            logging.error(f"Error in decay_memories: {e}")
            # Retry once after a short delay
            time.sleep(0.5)
            self.cursor.execute('''
                UPDATE agent_memories
                SET decay_factor = MAX(0, decay_factor - ? * 
                    (julianday('now') - julianday(last_accessed))),
                    last_accessed = datetime('now')
                WHERE user_id = ?
            ''', (self.memory_decay_rate, self.user_id))
    
    def _evaluate_memory_importance(self, content: str) -> float:
        """Evaluate the importance of a memory based on its content."""
        importance_factors = {
            'emotional_words': ['love', 'hate', 'angry', 'happy', 'sad'],
            'action_words': ['achieved', 'failed', 'learned', 'discovered'],
            'relationship_words': ['friend', 'follow', 'connect', 'share'],
            'goal_words': ['objective', 'target', 'aim', 'purpose']
        }
        
        score = 0.5  # Base score
        content_lower = content.lower()
        
        for category, words in importance_factors.items():
            for word in words:
                if word in content_lower:
                    score += 0.1
        
        return min(1.0, max(0.0, score)) 