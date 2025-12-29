from pydantic import BaseModel
from typing import List, Dict
from datetime import datetime

class StudentState(BaseModel):
    student_id: str
    
    # 1. Performance Vector: Score per topic (0.0 to 1.0)
    # Example: {"algebra": 0.6, "geometry": 0.8}
    topic_mastery: Dict[str, float]
    
    # 2. Cognitive Load Tracker
    # Based on: (actual_time_spent / expected_time)
    current_stress_level: float 
    
    # 3. Forgetting Curve Data
    # Key: Topic, Value: Timestamp of last successful recall
    last_interaction: Dict[str, datetime]
    
    # 4. Stability Index (S)
    # How many times they have successfully reviewed this in the past
    retention_strength: Dict[str, int]

    class Config:
        arbitrary_types_allowed = True