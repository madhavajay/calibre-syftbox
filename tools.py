import json
import os
from pathlib import Path
from typing import List

from pydantic import BaseModel

calibre_convert = "/Applications/calibre.app/Contents/MacOS/ebook-convert"


class Book(BaseModel):
    file_path: str
    title: str | None = None
    author: str | None = None
    cover_path: str | None = None
    text_path: str | None = None
    is_converted: bool = False

    def __init__(self, **data):
        super().__init__(**data)
        self.text_path = self.file_path.rsplit(".", 1)[0] + ".txt"
        self.cover_path = self.file_path.replace(
            self.file_path.split("/")[-1], "cover.jpg"
        )
        self.extract_author_and_title()
        self.is_converted = os.path.exists(self.text_path)

    def extract_author_and_title(self):
        # Split the file path into parts
        path_parts = self.file_path.split("/")

        # Assuming the structure is consistent, extract author and title
        if len(path_parts) >= 3:
            self.author = path_parts[-3]
            self.title = path_parts[-2]

    def convert_to_text(self, calibre_convert: str):
        if self.text_path is None or not os.path.exists(self.text_path):
            # Ensure the text_path is set to a valid output path
            self.text_path = self.file_path.rsplit(".", 1)[0] + ".txt"
            print(f"Converting {self.file_path} to text. Output path: {self.text_path}")
            # Use quotes around paths to handle spaces and special characters
            command = f'"{calibre_convert}" "{self.file_path}" "{self.text_path}"'
            print(command)
            os.system(command)
            print(f"Conversion complete for {self.file_path}")
            self.is_converted = True


def scan_calibre_library(library_path: str) -> List[Book]:
    """
    Scans a Calibre library directory for books with .epub, .mobi, and .pdf extensions.

    Args:
        library_path (str): The path to the Calibre library.

    Returns:
        List[Book]: A list of Book objects for books with the specified extensions.
    """
    library = Path(library_path)
    if not library.is_dir():
        raise ValueError(f"The path {library_path} is not a valid directory.")

    book_dict = {}
    for file in library.rglob("*"):
        if file.suffix in {".epub", ".mobi", ".pdf"}:
            book_name = file.stem
            current_format = file.suffix
            if book_name not in book_dict:
                book_dict[book_name] = (str(file), current_format)
            else:
                _, existing_format = book_dict[book_name]
                # Prioritize formats: .epub > .mobi > .pdf
                if (
                    existing_format == ".pdf" and current_format in {".epub", ".mobi"}
                ) or (existing_format == ".mobi" and current_format == ".epub"):
                    book_dict[book_name] = (str(file), current_format)

    books = [Book(file_path=path) for path, _ in book_dict.values()]
    return books


class Settings(BaseModel):
    calibre_convert_path: str = "/Applications/calibre.app/Contents/MacOS/ebook-convert"
    calibre_library_path: str = str(Path.home() / "Calibre Library")

    @classmethod
    def get_data_path(cls, app_data_dir: Path) -> Path:
        data_path = app_data_dir / "data"
        data_path.mkdir(parents=True, exist_ok=True)
        return data_path

    @classmethod
    def load(cls, app_data_dir: Path) -> "Settings":
        settings_path = cls.get_data_path(app_data_dir=app_data_dir) / "settings.json"
        if settings_path.exists():
            with settings_path.open("r") as config_file:
                try:
                    settings_data = json.load(config_file)
                    return cls(**settings_data)
                except json.JSONDecodeError:
                    print(
                        "Warning: settings.json is empty or corrupted. Loading default settings."
                    )
                    default_settings = cls()
                    default_settings.save(app_data_dir)
                    return default_settings
        else:
            # If the file does not exist, create it with default settings
            default_settings = cls()
            default_settings.save(app_data_dir)
            return default_settings

    def save(self, app_data_dir: Path):
        settings_path = self.get_data_path(app_data_dir=app_data_dir) / "settings.json"
        with settings_path.open("w") as config_file:
            json.dump(self.dict(), config_file, indent=4)
