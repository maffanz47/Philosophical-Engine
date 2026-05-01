from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from ..core.permissions import require_user
from ..config import settings
import anthropic
from typing import List, Dict

router = APIRouter()

class QueryRequest:
    message: str
    history: List[Dict[str, str]]
    context: Dict

@router.post("/query")
async def query_claude(request: QueryRequest, current_user = Depends(require_user)):
    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    
    system_prompt = f"""You are a philosophical tutor embedded in the Philosophical Engine, an AI analysis
tool for philosophy students. The student has just received these analysis results:
{request.context}
Help them understand what the results mean, explore the philosophical ideas deeper,
challenge their thinking in a Socratic way, suggest related philosophers or readings,
and answer any philosophy questions they have. Be intellectually rigorous but accessible.
Keep responses under 200 words unless asked for more detail."""
    
    messages = [{"role": "system", "content": system_prompt}] + request.history + [{"role": "user", "content": request.message}]
    
    async def generate():
        with client.messages.stream(
            model="claude-sonnet-4-20250514",
            max_tokens=1000,
            messages=messages
        ) as stream:
            for text in stream.text_stream:
                yield f"data: {text}\n\n"
    
    return StreamingResponse(generate(), media_type="text/event-stream")