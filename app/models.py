from sqlalchemy import Column, String, Integer, JSON, DateTime
from app.database import Base

class PrivateIdea(Base):
    __tablename__ = "private_ideas"

    id = Column(String, primary_key=True, index=True)
    idea_text = Column("idea", String)
    ownership_areas = Column(JSON)
    desired_traits = Column(JSON)
    idea_stage = Column(String)
    vision = Column(String)
    work_relationship = Column(JSON)
    pivot_openness = Column(Integer)
    matches = Column(JSON)
    # user_role = Column(String)
    created_at = Column(DateTime)
    updated_at = Column(DateTime)

class CofounderAnswer(Base):
    __tablename__ = "cofounder_answers"

    id = Column(String, primary_key=True, index=True)
    cofounder_id = Column(String)
    excited_areas = Column(JSON)
    ownership_areas = Column(JSON)
    personality_traits = Column(JSON)
    preferred_stage = Column(String)
    journey_type = Column(String)
    work_style = Column(JSON)
    suggestion_comfort = Column(Integer)