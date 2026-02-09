from app.core.groq_client import groq_client
import logging
import json
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

class BusinessProblemService:
    """
    Service for understanding and analyzing business problems
    Guides users through problem definition and solution strategy
    """
    
    _instance = None
    _business_context = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(BusinessProblemService, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, '_initialized'):
            self.groq = groq_client
            self._initialized = True
    
    def analyze_business_problem(self, user_description: str) -> Dict[str, Any]:
        """
        Analyze user's business problem and extract key information
        
        Args:
            user_description: User's description of their business problem
            
        Returns:
            Structured business context
        """
        try:
            logger.info(f"Analyzing business problem: {user_description[:100]}...")
            
            prompt = f"""You are a business analyst and data science consultant. A user has described their business problem.

User's Description:
"{user_description}"

Analyze this and extract the following information in JSON format:

1. problem_type: Categorize as one of:
   - "customer_churn" (predicting customer leaving)
   - "sales_forecasting" (predicting future sales)
   - "price_optimization" (finding optimal pricing)
   - "fraud_detection" (identifying fraudulent transactions)
   - "customer_segmentation" (grouping customers)
   - "demand_forecasting" (predicting product demand)
   - "lead_scoring" (ranking potential customers)
   - "sentiment_analysis" (analyzing customer feedback)
   - "recommendation_system" (product/content recommendations)
   - "inventory_optimization" (managing stock levels)
   - "risk_assessment" (evaluating risks)
   - "quality_prediction" (predicting product/service quality)
   - "other" (if none of above fit)

2. industry: The user's industry (e.g., "retail", "finance", "healthcare", "e-commerce", "manufacturing")

3. business_goal: The primary business objective (e.g., "reduce churn by 20%", "increase revenue")

4. success_metric: How success will be measured (e.g., "churn rate", "revenue", "accuracy")

5. target_variable: What needs to be predicted (e.g., "will_churn", "sales_amount", "fraud_flag")

6. key_questions: List 3-5 key questions to ask the user to better understand their problem

7. recommended_approach: Brief description of the recommended ML approach

8. data_requirements: List what data they should have

9. potential_challenges: List 2-3 potential challenges

Return ONLY valid JSON, no markdown or explanation.
"""

            messages = [{"role": "user", "content": prompt}]
            response = self.groq.chat_completion(messages, temperature=0.3)
            
            # Parse response
            analysis = self._parse_json_response(response)
            
            # Store in class variable for later use
            BusinessProblemService._business_context = analysis
            
            logger.info(f"Problem categorized as: {analysis.get('problem_type')}")
            
            return analysis
            
        except Exception as e:
            logger.error(f"Business problem analysis error: {str(e)}")
            raise
    
    def generate_clarifying_questions(self, analysis: Dict[str, Any]) -> List[str]:
        """
        Generate follow-up questions to better understand the business problem
        """
        return analysis.get('key_questions', [
            "What is your main business challenge?",
            "What data do you currently have available?",
            "What outcome are you trying to achieve?"
        ])
    
    def refine_with_answers(self, answers: Dict[str, str]) -> Dict[str, Any]:
        """
        Refine business context based on user's answers to questions
        """
        try:
            context = BusinessProblemService._business_context or {}
            
            prompt = f"""Based on the initial business problem analysis and user's answers to follow-up questions, 
refine and enhance the business context.

Initial Analysis:
{json.dumps(context, indent=2)}

User's Answers:
{json.dumps(answers, indent=2)}

Provide an updated and more detailed business context in JSON format with the same structure as before,
but with more specific and actionable information.

Return ONLY valid JSON.
"""

            messages = [{"role": "user", "content": prompt}]
            response = self.groq.chat_completion(messages, temperature=0.3)
            
            refined = self._parse_json_response(response)
            BusinessProblemService._business_context = refined
            
            return refined
            
        except Exception as e:
            logger.error(f"Refine context error: {str(e)}")
            return context
    
    def get_contextual_analysis_prompt(self, data_summary: Dict[str, Any]) -> str:
        """
        Generate analysis prompt based on business context
        """
        context = BusinessProblemService._business_context
        if not context:
            return None
        
        prompt = f"""Analyze this dataset in the context of the following business problem:

Business Context:
- Industry: {context.get('industry', 'Unknown')}
- Problem Type: {context.get('problem_type', 'Unknown')}
- Business Goal: {context.get('business_goal', 'Unknown')}
- Target Variable: {context.get('target_variable', 'Unknown')}

Dataset Summary:
- Rows: {data_summary.get('num_rows', 0)}
- Columns: {data_summary.get('num_columns', 0)}
- Available Columns: {', '.join(data_summary.get('columns', []))}

Provide insights specifically focused on:
1. How well this data supports the business goal
2. Which columns are most relevant for predicting {context.get('target_variable')}
3. What additional data might be needed
4. Potential issues or concerns for this specific use case
5. Recommended next steps

Be specific and actionable.
"""
        return prompt
    
    def get_business_context(self) -> Dict[str, Any]:
        """Get current business context"""
        return BusinessProblemService._business_context or {}
    
    def clear_context(self):
        """Clear stored business context"""
        BusinessProblemService._business_context = None
    
    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """Parse JSON from LLM response"""
        try:
            # Remove markdown formatting
            cleaned = response.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            if cleaned.startswith("```"):
                cleaned = cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()
            
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error: {str(e)}")
            return {
                "problem_type": "other",
                "industry": "unknown",
                "business_goal": "analyze data",
                "key_questions": ["What is your business goal?"],
                "recommended_approach": "exploratory analysis",
                "data_requirements": ["relevant data"],
                "potential_challenges": ["unclear requirements"]
            }
    
    def generate_problem_summary(self) -> str:
        """
        Generate a human-readable summary of the business problem
        """
        context = BusinessProblemService._business_context
        if not context:
            return "No business context defined yet."
        
        summary = f"""
📊 Business Problem Summary

🏢 Industry: {context.get('industry', 'Not specified')}
🎯 Problem Type: {context.get('problem_type', 'Not specified').replace('_', ' ').title()}
💡 Business Goal: {context.get('business_goal', 'Not specified')}
📈 Success Metric: {context.get('success_metric', 'Not specified')}
🎲 Target Variable: {context.get('target_variable', 'Not specified')}

🔍 Recommended Approach:
{context.get('recommended_approach', 'Not specified')}

📋 Data Requirements:
{chr(10).join(['- ' + req for req in context.get('data_requirements', ['Not specified'])])}

⚠️ Potential Challenges:
{chr(10).join(['- ' + challenge for challenge in context.get('potential_challenges', ['Not specified'])])}
"""
        return summary