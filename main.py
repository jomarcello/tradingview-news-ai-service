import os
import json
import logging
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_KEY"))

SYSTEM_PROMPT = """You are a financial news analyst. Analyze the provided news articles and create a concise but comprehensive market analysis. Focus on the potential impact on trading decisions.

Your analysis should follow this format:

 *Market Impact Analysis*

• ECB's latest decision: [key point]
• Market implications: [key point]
• Current trend: [key point]

 *Market Sentiment*

• Direction: [Bullish/Bearish/Neutral]
• Strength: [Strong/Moderate/Weak]
• Key driver: [One line explanation]

 *Trading Implications*

• Short-term outlook: [Expected impact]
• Risk assessment: [High/Medium/Low]
• Key levels: [Support/Resistance if relevant]

 *Risk Factors*

• [Risk factor 1]
• [Risk factor 2]
• [Risk factor 3]

Remember:
- Use clear, simple language
- Keep each point concise
- Add a space between sections
- Use * for emphasis instead of #
- Format numbers clearly (1.2350 instead of 1.235)"""

class NewsRequest(BaseModel):
    instrument: str
    articles: List[Dict[str, Any]]

@app.post("/analyze-news")
def analyze_news(request: NewsRequest):
    """Analyze news articles and provide sentiment analysis."""
    try:
        # Create a prompt for news analysis
        articles_text = "\n\n".join([
            f"Title: {article.get('title', 'No Title')}\n"
            f"Content: {article.get('content', 'No Content')}\n"
            f"Source: {article.get('source', 'Unknown')}\n"
            f"Date: {article.get('date', 'Unknown')}"
            for article in request.articles
        ])
        
        prompt = f"""Analyze these news articles about {request.instrument} and provide a concise market analysis.

News Articles:
{articles_text}

Format your response according to the following guidelines:
{SYSTEM_PROMPT}"""

        # Call OpenAI API for analysis
        analysis_response = client.chat.completions.create(
            model="gpt-4-0125-preview",
            messages=[{
                "role": "system",
                "content": SYSTEM_PROMPT
            }, {
                "role": "user",
                "content": prompt
            }],
            temperature=0.5,
            max_tokens=1000
        )
        
        # Get a specific trading verdict
        verdict_prompt = f"""Based on the news analysis, provide a clear trading verdict for {request.instrument}.
        Previous analysis: {analysis_response.choices[0].message.content}
        
        Format your response as a JSON with these fields:
        - verdict: (STRONG_BUY, BUY, NEUTRAL, SELL, STRONG_SELL)
        - confidence: (percentage between 0-100)
        - key_reason: (brief explanation)"""
        
        verdict_response = client.chat.completions.create(
            model="gpt-4-0125-preview",
            messages=[{
                "role": "system",
                "content": "You are a trading advisor that provides clear, decisive trading verdicts based on news analysis."
            }, {
                "role": "user",
                "content": verdict_prompt
            }],
            temperature=0.3,
            max_tokens=200
        )
        
        logger.info(f"Successfully analyzed news for {request.instrument}")
        
        try:
            verdict_json = json.loads(verdict_response.choices[0].message.content)
        except json.JSONDecodeError:
            # If the response is not valid JSON, create a structured response
            verdict_json = {
                "verdict": "NEUTRAL",
                "confidence": 50,
                "key_reason": "Could not parse verdict response"
            }
        
        # Parse the OpenAI response into sections
        ai_response = analysis_response.choices[0].message.content
        
        # Extract sections from AI response
        sections = {}
        current_section = None
        current_content = []
        
        for line in ai_response.split('\n'):
            if 'Market Impact Analysis' in line:
                current_section = 'market_impact'
                continue
            elif 'Market Sentiment' in line:
                sections['market_impact'] = '\n'.join(current_content).strip()
                current_section = 'market_sentiment'
                current_content = []
                continue
            elif 'Trading Implications' in line:
                sections['market_sentiment'] = '\n'.join(current_content).strip()
                current_section = 'trading_implications'
                current_content = []
                continue
            elif 'Risk Factors' in line:
                sections['trading_implications'] = '\n'.join(current_content).strip()
                current_section = 'risk_factors'
                current_content = []
                continue
            
            if current_section and line.strip():
                current_content.append(line.strip())
        
        if current_content:
            sections['risk_factors'] = '\n'.join(current_content).strip()

        # Format the analysis with sections and emojis
        analysis = f"""Based on recent news and market data for {request.instrument}:

🔮 Market Impact Analysis
{sections.get('market_impact', 'No market impact analysis available.')}

📊 Market Sentiment
{sections.get('market_sentiment', 'No market sentiment available.')}

💡 Trading Implications
{sections.get('trading_implications', 'No trading implications available.')}

⚠️ Risk Factors
{sections.get('risk_factors', 'No risk factors available.')}"""
        
        return {
            "status": "success",
            "analysis": analysis,
            "verdict": verdict_json
        }
            
    except Exception as e:
        logger.error(f"Error analyzing news: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/get-market-context")
def get_market_context(instrument: str):
    """Get broader market context and potential correlations."""
    try:
        prompt = f"""Provide a brief market context analysis for {instrument}. Consider:
        1. Related instruments and their performance
        2. Key market drivers
        3. Important technical levels
        4. Upcoming economic events that might impact the instrument
        
        Format your response in a clear, concise way."""

        response = client.chat.completions.create(
            model="gpt-4-0125-preview",
            messages=[{
                "role": "system",
                "content": "You are a market analyst providing context and correlation analysis for trading instruments."
            }, {
                "role": "user",
                "content": prompt
            }],
            temperature=0.5,
            max_tokens=500
        )
        
        logger.info(f"Successfully got market context for {instrument}")
        return {"market_context": response.choices[0].message.content}
            
    except Exception as e:
        logger.error(f"Error getting market context: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
