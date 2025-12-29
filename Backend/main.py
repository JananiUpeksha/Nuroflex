from fastapi import FastAPI
from app.schemas.student_state import StudentState
from app.core.memory import ReplayBuffer

app = FastAPI()

# Initialize a global memory buffer
# In a real production app, you might use Redis or a DB for this
memory = ReplayBuffer(capacity=5000)

@app.post("/log-experience")
async def log_experience(state: list, action: int, reward: float, next_state: list, done: bool):
    """
    Every time Priya finishes a task, the frontend calls this 
    to feed the AI's memory.
    """
    memory.add(state, action, reward, next_state, done)
    
    return {
        "status": "Experience Saved",
        "current_buffer_size": len(memory)
    }