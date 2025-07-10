from typing import List, Optional
from pydantic import BaseModel
from datetime import datetime
class MatchRequest(BaseModel):
    Id: str

class MatchResult(BaseModel):
    cofounder_id: str
    match_score: float

class PrivateIdeaOut(BaseModel):
    id: str
    idea_text: str
    ownership_areas: List[str]
    desired_traits: List[str]
    idea_stage: str
    vision: str
    work_relationship: List[str]
    pivot_openness: int
    matches: Optional[List[dict]]
    # user_role: str
    created_at: datetime
    updated_at: datetime

    class Config:
        orm_mode = True
