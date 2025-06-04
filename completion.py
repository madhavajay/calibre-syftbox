import time
import uuid
from pathlib import Path
from typing import List, Literal, Optional

from fastapi import Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastsyftbox import FastSyftBox
from pydantic import BaseModel
from starlette.responses import JSONResponse

# Define OpenAI-style message format
class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str


# Define the request schema
class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 1.0
    max_tokens: Optional[int] = 256


# Define the response schema
class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str


class ChatCompletionResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: dict


async def chat_completions(request: Request, body: ChatCompletionRequest):
    try:
        auth = request.headers.get("Authorization")

        # Check for Bearer token

        # if not auth or not auth.startswith("Bearer "):
        #     raise HTTPException(
        #         status_code=401, detail="Missing or invalid Authorization header"
        #     )

        # Extract last user message
        last_user_msg = next(
            (m.content for m in reversed(body.messages) if m.role == "user"), "Hi"
        )

        print("last_user_msg", last_user_msg)

        # Dummy echo response
        reply = f"You said: {last_user_msg}"
        result = ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex}",
            object="chat.completion",
            created=int(time.time()),
            model=body.model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=reply),
                    finish_reason="stop",
                )
            ],
            usage={"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
        )
        print("result", result)

        return result
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)},
        )


class MessageModel(BaseModel):
    message: str
    name: str | None = None


class ModelItem(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str


class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelItem]


async def list_models(request: Request):
    return ModelList(
        data=[
            ModelItem(
                id="gpt-3.5-turbo", created=int(time.time()), owned_by="your-org"
            ),
            ModelItem(
                id="openai/chatgpt-4o-latest",
                created=int(time.time()),
                owned_by="your-org",
            ),
            ModelItem(id="llama-3-8b", created=int(time.time()), owned_by="your-org"),
        ]
    )
