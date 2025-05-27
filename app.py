import json
import os
import sys
import traceback
from pathlib import Path

import jinja2
from fastapi import FastAPI
from fastapi.requests import Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastsyftbox.syftbox import Syftbox
from pydantic import BaseModel

from tools import Book, Settings, scan_calibre_library

app_name = Path(__file__).resolve().parent.name
app = FastAPI(title=app_name)
syftbox = Syftbox(app=app, name=app_name)

current_dir = Path(__file__).parent
app_data_dir = Path(syftbox.client.config.data_dir) / "private" / "app_data" / app_name
app_data_dir.mkdir(parents=True, exist_ok=True)

app.mount(
    "/css",
    StaticFiles(directory=current_dir / "assets" / "css"),
    name="css",
)
app.mount("/js", StaticFiles(directory=current_dir / "assets" / "js"), name="js")


@app.get("/", response_class=HTMLResponse)
def root():
    settings = Settings.load(app_data_dir=app_data_dir)
    results = scan_calibre_library(settings.calibre_library_path)
    template_path = current_dir / "assets" / "index.html"

    with open(template_path) as f:
        template_content = f.read()

    books_total = len(results)
    text_total = sum(1 for book in results if book.is_converted)
    indexed_total = 0

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
        print("image_full_path is within the Calibre Library directory")
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
        return JSONResponse(
            status_code=500,
            content={"detail": f"An error occurred: {str(e)}"},
        )


class MessageModel(BaseModel):
    message: str
    name: str | None = None


# Build your DTN RPC endpoints available on
# syft://{datasite}/app_data/{app_name}/rpc/endpoint
@syftbox.on_request("/hello")
def hello_handler(request: MessageModel):
    response = MessageModel(message=f"Hi {request.name}", name="Bob")
    return response.model_dump_json()


# Debug your RPC endpoints in the browser
syftbox.enable_debug_tool(
    endpoint="/hello",
    example_request=str(MessageModel(message="Hello!", name="Alice").model_dump_json()),
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
