from sentence_transformers import SentenceTransformer, util
import torch
import logging

logger = logging.getLogger(__name__)

# Load your custom SBERT model with error handling
try:
    model = SentenceTransformer("knowidea-sbert")
    logger.info("Successfully loaded knowidea-sbert model")
except Exception as e:
    logger.error(f"Failed to load knowidea-sbert model: {str(e)}")
    # Fallback to a default model
    logger.info("Falling back to default model: all-MiniLM-L6-v2")
    model = SentenceTransformer("all-MiniLM-L6-v2")

def encode_founder(founder):
    try:
        # Handle potential None values and empty lists
        ownership_areas = founder.ownership_areas or []
        desired_traits = founder.desired_traits or []
        work_relationship = founder.work_relationship or []
        
        # Ensure we have at least one ownership area
        ownership_text = ownership_areas[0] if ownership_areas else "Not specified"
        
        founder_text = f"""
Idea: {founder.idea_text or "Not specified"}
Looking for cofounder to own: {ownership_text}
Personality traits: {', '.join(desired_traits) if desired_traits else "Not specified"}
Stage: {founder.idea_stage or "Not specified"}
Vision: {founder.vision or "Not specified"}
Work style: {', '.join(work_relationship) if work_relationship else "Not specified"}
Open to pivots: {founder.pivot_openness if founder.pivot_openness is not None else "Not specified"}
""".strip()
        
        logger.info(f"Encoded founder text length: {len(founder_text)}")
        return founder_text
        
    except Exception as e:
        logger.error(f"Error in encode_founder: {str(e)}")
        raise

def encode_cofounder(cof):
    try:
        # Handle potential None values and empty lists
        excited_areas = cof.excited_areas or []
        ownership_areas = cof.ownership_areas or []
        personality_traits = cof.personality_traits or []
        work_style = cof.work_style or []
        
        cof_text = f"""
Interested in: {', '.join(excited_areas) if excited_areas else "Not specified"}
Open to own: {', '.join(ownership_areas) if ownership_areas else "Not specified"}
Personality traits: {', '.join(personality_traits) if personality_traits else "Not specified"}
Stage preference: {cof.preffered_stage or "Not specified"}
Vision: {cof.journey_type or "Not specified"}
Work style: {', '.join(work_style) if work_style else "Not specified"}
Comfort with pivots: {cof.suggestion_comfort if cof.suggestion_comfort is not None else "Not specified"}
""".strip()
        
        logger.info(f"Encoded cofounder text length: {len(cof_text)}")
        return cof_text
        
    except Exception as e:
        logger.error(f"Error in encode_cofounder: {str(e)}")
        raise

def compute_similarity(text1: str, text2: str) -> float:
    try:
        if not text1 or not text2:
            logger.warning("Empty text provided for similarity computation")
            return 0.0
        
        emb1 = model.encode(text1, convert_to_tensor=True)
        emb2 = model.encode(text2, convert_to_tensor=True)
        similarity = util.cos_sim(emb1, emb2).item()
        
        # Ensure similarity is between 0 and 1
        similarity = max(0, min(1, similarity))
        
        return round(similarity * 100, 2)
        
    except Exception as e:
        logger.error(f"Error in compute_similarity: {str(e)}")
        raise