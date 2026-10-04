from click import prompt
import json
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict

from ppro_bridge.audit_logger import (
    read_latest_audit_event,
    write_audit_event,
)
from ppro_bridge.browser.perplexity_client import PerplexityClient
from ppro_bridge.request_interpreter import (
    PromptInterpretationError,
    extract_current_user_request,
)

MODEL_ID = "perplexity-web-bridge"
MAX_PROMPT_CHARS = 20_000


@asynccontextmanager
async def lifespan(app: FastAPI):
    client = PerplexityClient()
    await client.start()
    app.state.client = client
    yield
    await client.stop()


app = FastAPI(
    title="PPRO Bridge",
    version="0.5.0",
    lifespan=lifespan,
)


class Message(BaseModel):
    model_config = ConfigDict(extra="allow")
    role: str
    content: str | list[Any] | None = None


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    model: str
    messages: list[Message]
    stream: bool = False


def sse_event(payload: dict | str) -> str:
    if isinstance(payload, str):
        return f"data: {payload}\n\n"

    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return f"data: {encoded}\n\n"


def completion_response(
    completion_id: str,
    created: int,
    model: str,
    answer: str,
) -> dict:
    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": answer,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    }


async def stream_response(
    client: PerplexityClient,
    prompt: str,
    completion_id: str,
    created: int,
    model: str,
) -> AsyncIterator[str]:
    started_at = time.perf_counter()
    browser = await client.browser_status()

    try:
        answer = await client.ask(prompt)
    except TimeoutError as error:
        write_audit_event(
            request_id=completion_id,
            browser=browser,
            prompt=prompt,
            stream=True,
            status="timeout",
            duration_ms=int((time.perf_counter() - started_at) * 1000),
            error=str(error),
        )
        yield sse_event(
            {
                "error": {
                    "message": str(error),
                    "type": "timeout_error",
                    "code": "timeout",
                }
            }
        )
        yield sse_event("[DONE]")
        return
    except Exception as error:
        write_audit_event(
            request_id=completion_id,
            browser=browser,
            prompt=prompt,
            stream=True,
            status="browser_error",
            duration_ms=int((time.perf_counter() - started_at) * 1000),
            error=str(error),
        )
        yield sse_event(
            {
                "error": {
                    "message": f"Falha no navegador: {error}",
                    "type": "browser_error",
                    "code": "browser_error",
                }
            }
        )
        yield sse_event("[DONE]")
        return

    browser_after = await client.browser_status()

    write_audit_event(
        request_id=completion_id,
        browser=browser_after,
        prompt=prompt,
        stream=True,
        status="ok",
        duration_ms=int((time.perf_counter() - started_at) * 1000),
    )
    yield sse_event(
        {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant"},
                    "finish_reason": None,
                }
            ],
        }
    )

    if answer:
        yield sse_event(
            {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": answer},
                        "finish_reason": None,
                    }
                ],
            }
        )

    yield sse_event(
        {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop",
                }
            ],
        }
    )

    yield sse_event("[DONE]")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "mode": "production"}


@app.get("/debug/browser")
async def debug_browser(request: Request) -> dict:
    return await request.app.state.client.browser_status()


@app.get("/debug/audit/latest")
def debug_latest_audit() -> dict:
    event = read_latest_audit_event()

    if event is None:
        raise HTTPException(
            status_code=404,
            detail="Nenhum evento de auditoria foi registrado ainda.",
        )

    return event


@app.get("/v1/models")
def list_models() -> dict:
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_ID,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "ppro-bridge",
            }
        ],
    }


@app.post("/v1/chat/completions")
@app.post("/chat/completions")
async def chat_completions(body: ChatRequest, request: Request):
    if body.model != MODEL_ID:
        raise HTTPException(
            status_code=400,
            detail=f"Modelo suportado: {MODEL_ID}",
        )

    payload = body.model_dump(mode="json")

    try:
        prompt = extract_current_user_request(payload)
    except PromptInterpretationError as error:
        raise HTTPException(status_code=400, detail=str(error))

    if len(prompt) > MAX_PROMPT_CHARS:
        raise HTTPException(
            status_code=413,
            detail="Prompt limpo acima do limite.",
        )

    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    client = request.app.state.client

    if body.stream:
        return StreamingResponse(
            stream_response(
                client=client,
                prompt=prompt,
                completion_id=completion_id,
                created=created,
                model=body.model,
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    started_at = time.perf_counter()
    browser = await client.browser_status()

    try:
        answer = await client.ask(prompt)
    except TimeoutError as error:
        write_audit_event(
            request_id=completion_id,
            browser=browser,
            prompt=prompt,
            stream=False,
            status="timeout",
            duration_ms=int((time.perf_counter() - started_at) * 1000),
            error=str(error),
        )
        raise HTTPException(status_code=504, detail=str(error))
    except Exception as error:
        write_audit_event(
            request_id=completion_id,
            browser=browser,
            prompt=prompt,
            stream=False,
            status="browser_error",
            duration_ms=int((time.perf_counter() - started_at) * 1000),
            error=str(error),
        )
        raise HTTPException(
            status_code=502,
            detail=f"Falha no navegador: {error}",
        )

    write_audit_event(
        request_id=completion_id,
        browser=browser,
        prompt=prompt,
        stream=False,
        status="ok",
        duration_ms=int((time.perf_counter() - started_at) * 1000),
    )

    return completion_response(
        completion_id=completion_id,
        created=created,
        model=body.model,
        answer=answer,
    )
