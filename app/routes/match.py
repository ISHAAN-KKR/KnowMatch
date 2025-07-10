# from fastapi import APIRouter, Depends, HTTPException
# from sqlalchemy.orm import Session
# from app.database import SessionLocal
# from app.models import PrivateIdea, CofounderAnswer
# from app.embeddings import encode_founder, encode_cofounder, compute_similarity
# from app.schemas import MatchRequest, MatchResult
# import logging
# import traceback

# # Set up logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# router = APIRouter()

# def get_db():
#     db = SessionLocal()
#     try:
#         yield db
#     finally:
#         db.close()

# @router.post("/match", response_model=list[MatchResult])
# def match_founder(request: MatchRequest, db: Session = Depends(get_db)):
#     try:
#         logger.info(f"Received match request for ID: {request.Id}")
        
#         # Query founder
#         founder = db.query(PrivateIdea).filter(PrivateIdea.id == request.Id).first()
        
#         if not founder:
#             logger.error(f"Founder with ID {request.Id} not found")
#             raise HTTPException(status_code=404, detail="Founder idea not found.")
        
#         logger.info(f"Found founder: {founder.id}")
        
#         # Check if founder has required fields
#         if not founder.idea_text:
#             logger.error(f"Founder {founder.id} has no idea_text")
#             raise HTTPException(status_code=400, detail="Founder idea has no text")
        
#         # Encode founder
#         try:
#             founder_text = encode_founder(founder)
#             logger.info(f"Encoded founder text: {founder_text[:100]}...")
#         except Exception as e:
#             logger.error(f"Error encoding founder: {str(e)}")
#             raise HTTPException(status_code=500, detail=f"Error encoding founder: {str(e)}")
        
#         # Get cofounders
#         cofounders = db.query(CofounderAnswer).all()
#         logger.info(f"Found {len(cofounders)} cofounders")
        
#         if not cofounders:
#             logger.warning("No cofounders found in database")
#             return []
        
#         results = []
#         for cof in cofounders:
#             try:
#                 if not cof.excited_areas or not cof.ownership_areas:
#                     logger.warning(f"Cofounder {cof.cofounder_id} missing required fields")
#                     continue
                
#                 cof_text = encode_cofounder(cof)
#                 score = compute_similarity(founder_text, cof_text)
#                 results.append({
#                     "cofounder_id": cof.cofounder_id,
#                     "match_score": round(score, 2)
#                 })
#                 logger.info(f"Matched with cofounder {cof.cofounder_id}: score {score}")
#             except Exception as e:
#                 logger.error(f"Error processing cofounder {cof.cofounder_id}: {str(e)}")
#                 continue
        
#         if not results:
#             logger.warning("No valid matches found")
#             return []
        
#         # Sort and save top 5 matches
#         top_matches = sorted(results, key=lambda x: x["match_score"], reverse=True)[:5]
#         logger.info(f"Returning {len(top_matches)} top matches")

#         founder.matches = [
#             {
#                 "cofounderId": match["cofounder_id"],
#                 "matchPercentage": match["match_score"]
#             } for match in top_matches
#         ]
#         db.commit()
#         logger.info(f"Saved matches to founder {founder.id}")

#         return top_matches
        
#     except HTTPException:
#         raise
#     except Exception as e:
#         logger.error(f"Unexpected error in match_founder: {str(e)}")
#         traceback.print_exc()
#         raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
# @router.post("/idea/details")
# def get_founder_details(request: MatchRequest, db: Session = Depends(get_db)):
#    """Get all details for a specific founder by ID via POST"""
#    try:
#        founder = db.query(PrivateIdea).filter(PrivateIdea.id == request.Id).first()
       
#        if not founder:
#            raise HTTPException(status_code=404, detail="Founder not found")
       
#        return {
#            "id": founder.id,
#            "idea_text": founder.idea_text,
#            "ownership_areas": founder.ownership_areas,
#            "desired_traits": founder.desired_traits,
#            "idea_stage": founder.idea_stage,
#            "vision": founder.vision,
#            "work_relationship": founder.work_relationship,
#            "pivot_openness": founder.pivot_openness,
#            "matches": founder.matches,
#            "created_at": founder.created_at,
#            "updated_at": founder.updated_at
#        }
       
#    except HTTPException:
#        raise
#    except Exception as e:
#        logger.error(f"Error fetching founder details: {str(e)}")
#        raise HTTPException(status_code=500, detail=str(e))

# @router.get("/cofounders")
# def get_all_cofounders(db: Session = Depends(get_db)):
#    """Get all cofounder IDs and basic info via GET"""
#    try:
#        cofounders = db.query(CofounderAnswer).all()
       
#        result = []
#        for cof in cofounders:
#            result.append({
#                "id": cof.id,
#                "cofounder_id": cof.cofounder_id,
#                "excited_areas": cof.excited_areas,
#                "ownership_areas": cof.ownership_areas,
#                "personality_traits": cof.personality_traits,
#                "preferred_stage": cof.preferred_stage,
#                "journey_type": cof.journey_type,
#                "work_style": cof.work_style,
#                "suggestion_comfort": cof.suggestion_comfort
#            })
       
#        return {
#            "total_cofounders": len(result),
#            "cofounders": result
#        }
       
#    except Exception as e:
#        logger.error(f"Error fetching cofounders: {str(e)}")
#        raise HTTPException(status_code=500, detail=str(e))
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.database import SessionLocal
from app.models import PrivateIdea, CofounderAnswer
from app.embeddings import encode_founder, encode_cofounder, compute_similarity
from app.schemas import MatchRequest
import logging
import traceback

# Set up logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

# Dependency for DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.post("/match")
def match_founder_all(request: MatchRequest, db: Session = Depends(get_db)):
    try:
        logger.info(f"Received match request for founder ID: {request.Id}")

        founder = db.query(PrivateIdea).filter(PrivateIdea.id == request.Id).first()
        if not founder:
            logger.error(f"Founder with ID {request.Id} not found")
            raise HTTPException(status_code=404, detail="Founder not found")

        # Encode founder
        try:
            founder_text = encode_founder(founder)
        except Exception as e:
            logger.error(f"Error encoding founder: {e}")
            raise HTTPException(status_code=500, detail="Error encoding founder")

        cofounders = db.query(CofounderAnswer).all()
        results = []

        for cof in cofounders:
            try:
                # Encode cofounder
                cof_text = encode_cofounder(cof)
                score = compute_similarity(founder_text, cof_text)

                logger.info(f"Match score with {cof.cofounder_id}: {score}")

                results.append({
                    "id": cof.id,
                    "cofounder_id": cof.cofounder_id,
                    "excited_areas": cof.excited_areas,
                    "ownership_areas": cof.ownership_areas,
                    "personality_traits": cof.personality_traits,
                    "preferred_stage": cof.preferred_stage,
                    "journey_type": cof.journey_type,
                    "work_style": cof.work_style,
                    "suggestion_comfort": cof.suggestion_comfort,
                    "match_score": round(score, 2)
                })
            except Exception as e:
                logger.warning(f"Skipping cofounder {cof.cofounder_id}: {str(e)}")
                traceback.print_exc()
                continue

        return {
            "founder": {
                "id": founder.id,
                "idea_text": founder.idea_text,
                "ownership_areas": founder.ownership_areas,
                "desired_traits": founder.desired_traits,
                "idea_stage": founder.idea_stage,
                "vision": founder.vision,
                "work_relationship": founder.work_relationship,
                "pivot_openness": founder.pivot_openness,
                "matches": founder.matches,
                "created_at": founder.created_at,
                "updated_at": founder.updated_at
            },
            "total_cofounders": len(results),
            "cofounders": results
        }

    except Exception as e:
        logger.error(f"Internal error: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Internal server error")
