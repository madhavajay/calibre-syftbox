import time
import json
import uuid
import json
import os
import sys
import traceback
from pathlib import Path

import jinja2
from fastapi.requests import Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.responses import Response
from fastapi import FastAPI
from contextlib import asynccontextmanager
import asyncio
from fastsyftbox import FastSyftBox
from pydantic import BaseModel
from completion import ChatCompletionResponse, ChatCompletionChoice, ChatMessage

from rag_index import RAGIndexer, create_background_indexer_loop
from tools import Book, Settings, scan_calibre_library
from syft_core import Client as SyftboxClient
from syft_core import SyftClientConfig
from completion import ChatCompletionResponse, ModelList, chat_completions, list_models, ChatCompletionRequest, ChatMessage

app_name = Path(__file__).resolve().parent.name

config = SyftClientConfig.load()
client = SyftboxClient(config)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Start background task
    task = asyncio.create_task(create_background_indexer_loop(app_data_dir=app_data_dir)())
    print("Background task started.")
    yield
    # Shutdown logic
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        print("Background task cancelled.")

app = FastSyftBox(
    app_name=app_name,
    syftbox_config=config,
    lifespan=lifespan,
    syftbox_endpoint_tags=[
        "syftbox"
    ],  # endpoints with this tag are also available via Syft RPC
    include_syft_openapi=True,  # Create OpenAPI endpoints for syft-rpc routes
)

current_dir = Path(__file__).parent
app_data_dir = Path(client.config.data_dir) / "private" / "app_data" / app_name
app_data_dir.mkdir(parents=True, exist_ok=True)


app.mount(
    "/css",
    StaticFiles(directory=current_dir / "assets" / "css"),
    name="css",
)
app.mount("/js", StaticFiles(directory=current_dir / "assets" / "js"), name="js")

indexer = RAGIndexer(app_data_dir=app_data_dir)


@app.get("/", response_class=HTMLResponse)
def root():
    settings = Settings.load(app_data_dir=app_data_dir)
    results = scan_calibre_library(settings.calibre_library_path)
    template_path = current_dir / "assets" / "index.html"

    with open(template_path) as f:
        template_content = f.read()

    books_total = len(results)
    text_total = sum(1 for book in results if book.is_converted)
    indexed_total = indexer.count_indexed_books()

    render_context = {
        "library_stats": {
            "books_total": books_total,
            "text_total": text_total,
            "indexed_total": indexed_total,
        }
    }
    print("render_context", render_context)

    template = jinja2.Template(template_content)
    rendered_content = template.render(**render_context)
    return rendered_content


@app.get("/index_book", response_class=JSONResponse)
async def index_book(request: Request):
    try:
        file_path = request.query_params.get("path")

        if not file_path or not Path(file_path).exists():
            return JSONResponse(
                status_code=400,
                content={"detail": "Invalid or missing file path."},
            )

        # Load settings to get the Calibre library path
        settings = Settings.load(app_data_dir=app_data_dir)

        # Construct the Book object
        if not file_path.startswith(settings.calibre_library_path):
            return JSONResponse(
                status_code=400,
                content={"detail": "File path is not within the Calibre library path."},
            )
        book = Book(file_path=file_path)

        # Add the book to the index
        indexer.add_book_to_index(book)

        return JSONResponse(
            status_code=200,
            content={"detail": f"Book at {file_path} indexed successfully."},
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"detail": f"An error occurred: {str(e)}"},
        )


@app.get("/settings", response_class=HTMLResponse)
def get_settings():
    settings = Settings.load(app_data_dir=app_data_dir)
    template_path = current_dir / "assets" / "settings.html"

    with open(template_path) as f:
        template_content = f.read()

    template = jinja2.Template(template_content)
    rendered_content = template.render(settings=settings.model_dump())
    return rendered_content


@app.post("/settings", response_class=JSONResponse)
async def post_settings(request: Request):
    try:
        # Decode the JSON payload from the request
        payload = await request.json()
        print("payload", payload)

        # Extract the required keys from the payload
        calibre_convert_path = payload.get(
            "calibre_convert_path",
        )
        calibre_library_path = payload.get(
            "calibre_library_path",
        )

        # Load and update settings
        settings = Settings.load(app_data_dir=app_data_dir)
        invalid_paths = []

        if not Path(calibre_convert_path).exists():
            invalid_paths.append("Calibre Convert Binary Path")

        if not Path(calibre_library_path).exists():
            invalid_paths.append("Calibre Library Folder Path")

        if invalid_paths:
            return JSONResponse(
                status_code=400,
                content={"detail": f"Invalid paths: {', '.join(invalid_paths)}"},
            )

        settings.calibre_convert_path = calibre_convert_path
        settings.calibre_library_path = calibre_library_path
        settings.save(app_data_dir=app_data_dir)

        return JSONResponse(
            status_code=200, content={"detail": "Settings saved successfully"}
        )
    except Exception as e:
        return JSONResponse(
            status_code=500, content={"detail": f"An error occurred: {str(e)}"}
        )


@app.get("/library", response_class=HTMLResponse)
def library():
    settings = Settings.load(app_data_dir=app_data_dir)
    results = scan_calibre_library(settings.calibre_library_path)
    template_path = current_dir / "assets" / "library.html"

    with open(template_path) as f:
        template_content = f.read()

    render_context = {
        "books": json.dumps([book.model_dump() for book in results]),
    }
    template = jinja2.Template(template_content)
    rendered_content = template.render(**render_context)
    return rendered_content


@app.get("/get_image", response_class=HTMLResponse)
async def get_image(request: Request):
    settings = Settings.load(app_data_dir=app_data_dir)
    image_path = request.query_params.get("path")
    image_full_path = Path(image_path).resolve()

    # Check if the image path is within the Calibre Library directory
    if (
        str(settings.calibre_library_path) in str(image_full_path.parents[0])
        and image_full_path.suffix == ".jpg"
    ):
        try:
            with open(image_full_path, "rb") as image_file:
                return HTMLResponse(content=image_file.read(), media_type="image/jpeg")
        except FileNotFoundError:
            return JSONResponse(
                status_code=404,
                content={"detail": "Image not found"},
            )
    else:
        return JSONResponse(
            status_code=400,
            content={"detail": "Invalid image path or not a .jpg file"},
        )

async def query_books(query: str):
    settings = Settings.load(app_data_dir=app_data_dir)
    try:

        if not query:
            return JSONResponse(
                status_code=400,
                content={"detail": "Query is required."},
            )

        results = indexer.search_index(query)

        if not results:
            return JSONResponse(
                status_code=200,
                content={
                    "answer": "No specific answer found, but here are some relevant excerpts from your books.",
                    "sources": [],
                },
            )

        # Format the results
        formatted_results = []
        book_cache = {}
        for result in results:
            book_hash = result["book_hash"]
            print(book_hash)
            if book_hash not in book_cache:
                book_cache[book_hash] = Book.get_by_hash(
                    book_hash, settings.calibre_library_path
                )
            book = book_cache[book_hash]
            print("got book", book)
            formatted_results.append(
                {
                    "book_title": book.title,
                    "author": book.author,
                    "excerpt": result["chunk"],
                }
            )

        return JSONResponse(
            status_code=200,
            content={
                "answer": "Here are some relevant excerpts from your books.",
                "sources": formatted_results,
            },
        )

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"detail": f"An error occurred: {str(e)}"},
        )


@app.post("/query_books", response_class=JSONResponse)
async def query_books_endpoint(request: Request):
    try:
        data = await request.json()
        query = data.get("query")
        return await query_books(query)
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"detail": f"An error occurred: {str(e)}"},
        )

@app.post("/convert_book", response_class=JSONResponse)
async def convert_book(request: Request):
    settings = Settings.load(app_data_dir=app_data_dir)
    try:
        data = await request.json()
        file_path = data.get("file_path")

        if not file_path:
            return JSONResponse(
                status_code=400,
                content={"detail": "File path is required."},
            )

        # Recreate the Book object
        book = Book(file_path=file_path)

        # Check if the book is already converted
        if book.is_converted:
            return JSONResponse(
                status_code=200,
                content={"detail": "Book is already converted."},
            )

        # Call the convert_to_text method
        book.convert_to_text(settings.calibre_convert_path)

        return JSONResponse(
            status_code=200,
            content={
                "detail": "Book conversion successful.",
                "text_path": book.text_path,
            },
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"detail": f"An error occurred: {str(e)}"},
        )


@app.post("/remove_book", response_class=JSONResponse)
async def remove_book(request: Request):
    settings = Settings.load(app_data_dir=app_data_dir)
    try:
        data = await request.json()
        file_path = data.get("file_path")

        if not file_path:
            return JSONResponse(
                status_code=400,
                content={"detail": "File path is required."},
            )

        # Ensure the file path is within the library path
        library_path = Path(settings.calibre_library_path)
        target_path = Path(file_path).with_suffix(".txt")

        if not target_path.is_relative_to(library_path):
            return JSONResponse(
                status_code=400,
                content={"detail": "File path is not within the library path."},
            )

        book = Book(file_path=file_path)
        indexer.remove_book_from_index(book)

        # Remove the .txt file
        if target_path.exists():
            os.remove(target_path)
            return JSONResponse(
                status_code=200,
                content={"detail": "Book text file removed successfully."},
            )
        else:
            return JSONResponse(
                status_code=404,
                content={"detail": "Text file not found."},
            )
    except Exception as e:
        import traceback

        print("error", traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={"detail": f"An error occurred: {str(e)}"},
        )



@app.post(
    "/v1/chat/completions", response_model=ChatCompletionResponse, tags=["syftbox"]
)
async def chat_completions_endpoint(request: Request, body: ChatCompletionRequest):
    try:
        last_user_msg = next(
            (m.content for m in reversed(body.messages) if m.role == "user"), "Hi"
        )
        query = last_user_msg
        print("last_user_msg", last_user_msg)
        response = await query_books(query)
        print("results", response)
        json_results = response.body.decode()

        print("json_results", json_results)
        result = ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex}",
            object="chat.completion",
            created=int(time.time()),
            model=body.model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=json_results),
                    finish_reason="stop",
                )
            ],
            usage={"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
        )
        print("result", result, type(result))
        return result
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"detail": f"An error occurred: {str(e)}"},
        )

@app.get("/v1/models", response_model=ModelList)
async def list_models_endpoint(request: Request):
    return await list_models()


class MessageModel(BaseModel):
    message: str
    name: str | None = None


# Build your DTN RPC endpoints available on
# syft://{datasite}/app_data/{app_name}/rpc/endpoint
@app.post("/hello", tags=["syftbox"])
def hello_handler(request: MessageModel) -> JSONResponse:
    response = MessageModel(message=f"Hi {request.name}", name="Bob")
    return response.model_dump_json()


example_request = str(
    ChatCompletionRequest(
        model="openai/chatgpt-4o-latest",
        messages=[ChatMessage(role="user", content="Hello!")],
        temperature=1.0,
        max_tokens=256,
    ).model_dump_json()
)

# Debug your Syft RPC endpoints in the browser
app.enable_debug_tool(
    endpoint="/v1/chat/completions",
    example_request=str(example_request),
    publish=True,
)


@app.exception_handler(Exception)
async def custom_exception_handler(request: Request, exc: Exception):
    # Get traceback object and format it
    tb = traceback.TracebackException.from_exception(exc)
    filtered_trace = [
        line
        for line in tb.format()
        if "/site-packages/"
        not in line  # filter out FastAPI, Starlette, AnyIO internals
    ]
    # Print just the relevant lines from your code
    print("".join(filtered_trace), file=sys.stderr)

    return JSONResponse(
        status_code=500,
        content={"detail": "Internal Server Error"},
    )


