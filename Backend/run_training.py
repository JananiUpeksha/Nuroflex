import numpy as np
import time
from app.core.trainer import NeuroTrainer
from app.core.memory import ReplayBuffer
from app.core.reward_system import calculate_reward

# 1. Setup Parameters
EPISODES = 1000      # How many "weeks" the AI will practice
BATCH_SIZE = 32      # How many memories to learn from at once
STATE_SIZE = 6       # [Mastery, Speed, Connectivity, Resilience, Stress, Retention]
ACTION_SIZE = 4      # [Review, Sprint, Challenge, Break]

trainer = NeuroTrainer(state_size=STATE_SIZE, action_size=ACTION_SIZE)
memory = ReplayBuffer(capacity=10000)

print("Starting Neuro-Training Phase...")

for e in range(EPISODES):
    # Start a fresh "Student Week"
    state = np.random.rand(STATE_SIZE) # Random initial student state
    total_reward = 0
    
    for day in range(7): # Simulating a 7-day plan
        # AI chooses an action (Explore or Exploit)
        action = trainer.get_action(state)
        
        # Simulate the result (In Phase 3, this comes from EdNet)
        # For now, we simulate a "Next State" based on the action
        next_state = state * 1.05 if action != 3 else state * 0.95
        reward = calculate_reward(state, next_state)
        
        done = True if day == 6 else False
        
        # Save the experience to memory
        memory.add(state, action, reward, next_state, done)
        
        # The Brain learns from its history
        trainer.train_step(memory, BATCH_SIZE)
        
        state = next_state
        total_reward += reward
        
    if e % 50 == 0:
        print(f"Episode: {e}/{EPISODES} | Total Reward: {total_reward:.2f} | Epsilon: {trainer.epsilon:.2f}")

print("Training Complete! The Brain is now ready for Phase 3.")
# At the very end of run_training.py
trainer.brain.save("neuro_brain_v1.h5")
print("Brain saved successfully as neuro_brain_v1.h5!")