from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.services.business_problem_service import BusinessProblemService
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

business_service = BusinessProblemService()

class ProblemDescription(BaseModel):
    description: str

class AnswersRequest(BaseModel):
    answers: Dict[str, str]

@router.post("/analyze")
async def analyze_business_problem(request: ProblemDescription):
    """
    Analyze user's business problem description
    """
    try:
        logger.info("Analyzing business problem")
        
        analysis = business_service.analyze_business_problem(request.description)
        questions = business_service.generate_clarifying_questions(analysis)
        
        return {
            "analysis": analysis,
            "questions": questions,
            "success": True
        }
        
    except Exception as e:
        logger.error(f"Business problem analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/refine")
async def refine_business_context(request: AnswersRequest):
    """
    Refine business context based on user answers
    """
    try:
        refined = business_service.refine_with_answers(request.answers)
        summary = business_service.generate_problem_summary()
        
        return {
            "context": refined,
            "summary": summary,
            "success": True
        }
        
    except Exception as e:
        logger.error(f"Refine context error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/context")
async def get_business_context():
    """
    Get current business context
    """
    try:
        context = business_service.get_business_context()
        summary = business_service.generate_problem_summary()
        
        return {
            "context": context,
            "summary": summary,
            "success": True
        }
        
    except Exception as e:
        logger.error(f"Get context error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/context")
async def clear_business_context():
    """
    Clear business context (start over)
    """
    try:
        business_service.clear_context()
        return {"success": True, "message": "Business context cleared"}
        
    except Exception as e:
        logger.error(f"Clear context error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))