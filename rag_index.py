import os
from tqdm import tqdm
import asyncio
import hashlib
import torch
import polars as pl
from datetime import datetime, timedelta
from pathlib import Path
from typing import List

import faiss
from multiprocessing import Process
import nltk
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from filelock import FileLock

from tools import Book, Settings, scan_calibre_library, book_hash

# Ensure punkt is downloaded
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)




class RAGIndexer:
    INDEX_FILENAME = "rag.index"
    META_FILENAME = "rag.meta.parquet"
    LOCK_FILENAME = "rag.lock"

    def __init__(self, app_data_dir: Path):
        self.app_data_dir = app_data_dir
        self.data_path = self.get_data_path(app_data_dir)
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        print("using device", device)
        self.model = SentenceTransformer("all-MiniLM-L6-v2", device=device)

    def search_index(self, query: str, top_k: int = 5):
        index, metadata = self.load_index_from_disk()

        if index.ntotal == 0:
            print("Index is empty.")
            return []

        # Encode the query
        query_embedding = self.model.encode([query], convert_to_numpy=True)

        # Perform search
        distances, indices = index.search(query_embedding, top_k)

        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx < len(metadata):
                row = metadata.row(idx)
                results.append({
                    "score": float(dist),
                    "chunk": row[1],
                    "book_hash": row[2]
                })
            else:
                print(f"[WARNING] Index {idx} out of metadata range.")

        return results


    @staticmethod
    def get_data_path(app_data_dir: Path) -> Path:
        data_path = app_data_dir / "data"
        data_path.mkdir(parents=True, exist_ok=True)
        return data_path

    @staticmethod
    def chunk_text(text: str, max_tokens=500, overlap=50) -> List[str]:
        sentences = sent_tokenize(text)
        chunks = []
        chunk = []
        token_count = 0

        for sentence in sentences:
            tokens = sentence.split()
            if token_count + len(tokens) > max_tokens:
                chunks.append(" ".join(chunk))
                chunk = chunk[-overlap:]
                token_count = sum(len(s.split()) for s in chunk)
            chunk.append(sentence)
            token_count += len(tokens)

        if chunk:
            chunks.append(" ".join(chunk))
        return chunks

    def load_index_from_disk(self):
        index_path = self.data_path / self.INDEX_FILENAME
        meta_path = self.data_path / self.META_FILENAME

        if index_path.exists():
            index = faiss.read_index(str(index_path))
        else:
            index = faiss.IndexFlatL2(384)

        if meta_path.exists():
            metadata = pl.read_parquet(meta_path)
        else:
            # Define schema explicitly to avoid SchemaError on vstack
            metadata = pl.DataFrame(
                schema={"id": pl.Int64, "chunk": pl.String, "book_hash": pl.String}
            )

        return index, metadata


    def save_index_to_disk(self, index, metadata: pl.DataFrame):
        index_path = self.data_path / self.INDEX_FILENAME
        meta_path = self.data_path / self.META_FILENAME
        print("waiting for lock")
        with FileLock(str(self.data_path / self.LOCK_FILENAME)):
            print("lock achieved")
            faiss.write_index(index, str(index_path))
            metadata.write_parquet(meta_path)

    def add_book_to_index(self, book: Book):
        base_path = Path(book.text_path)
        indexing_path = base_path.with_suffix(base_path.suffix + ".indexing")
        indexing_path.touch()

        log_path = Path(book.text_path).with_suffix(".rag.log")
        with open(log_path, "a") as log:
            start_time = datetime.now()
            message = f"[{start_time}] Starting indexing for {book.title}\n"
            log.write(message)
            print(message)

            if not book.text_path or not os.path.exists(book.text_path):
                message = f"Missing text for {book.title} at {book.text_path}\n"
                log.write(message)
                print(message)
                return

            indexed_path = base_path.with_suffix(base_path.suffix + ".indexed")

            try:
                index, metadata = self.load_index_from_disk()
                b_hash = book_hash(book.text_path)
                if b_hash in metadata["book_hash"].to_list():
                    message = f"Already indexed: {book.title}\n"
                    log.write(message)
                    print(message)
                    indexed_path.touch()
                    indexing_path.unlink(missing_ok=True)
                    return

                with open(book.text_path, "r", encoding="utf-8") as f:
                    raw_text = f.read()

                chunks = self.chunk_text(raw_text)
                message = f"Chunked into {len(chunks)} chunks\n"
                log.write(message)
                print(message)
                message = f"Encoding chunks...\n"
                log.write(message)
                print(message)

                embeddings = self.model.encode(
                    chunks, convert_to_numpy=True, batch_size=128, show_progress_bar=True
                )

                base_id = len(metadata)
                new_rows = [
                    {"id": base_id + i, "chunk": chunk, "book_hash": b_hash}
                    for i, chunk in enumerate(chunks)
                ]

                metadata = metadata.vstack(pl.DataFrame(new_rows))

                message = "Adding to FAISS index...\n"
                log.write(message)
                print(message)

                # Add in small chunks to monitor progress
                batch_size = 512
                for i in tqdm(range(0, len(embeddings), batch_size), desc="Adding to FAISS index"):
                    end = i + batch_size
                    index.add(embeddings[i:end])

                self.save_index_to_disk(index, metadata)
                indexed_path.touch()

                message = f"Completed indexing for {book.title}\n"
                log.write(message)
                print(message)

            finally:
                indexing_path.unlink(missing_ok=True)
                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds()
                message = f"[{end_time}] Finished in {duration:.2f} seconds\n"
                log.write(message)
                print(message)

    def count_indexed_books(self) -> int:
        _, metadata = self.load_index_from_disk()
        return len(set(metadata["book_hash"].to_list()))


def run_indexer(app_data_dir: Path, book: Book):
    indexer = RAGIndexer(app_data_dir=app_data_dir)
    indexer.add_book_to_index(book)


def spawn_indexer_process(app_data_dir: Path, book: Book):
    Process(target=run_indexer, args=(app_data_dir, book)).start()


def is_book_indexed(book: Book) -> bool:
    if not book.text_path:
        return False

    base_path = Path(book.text_path)
    indexing_path = base_path.with_suffix(base_path.suffix + ".indexing")
    indexed_path = base_path.with_suffix(base_path.suffix + ".indexed")

    if indexing_path.exists():
        mtime = datetime.fromtimestamp(indexing_path.stat().st_mtime)
        if datetime.now() - mtime < timedelta(minutes=10):
            # print(f"[DEBUG] Skipping {book.title}: indexing in progress")
            return True
        else:
            print(f"[DEBUG] Removing stale .indexing for {book.title}")
            indexing_path.unlink(missing_ok=True)

    if indexed_path.exists():
        # print(f"[DEBUG] Skipping {book.title}: already indexed (marker file found)")
        return True

    return False


def create_background_indexer_loop(app_data_dir, max_concurrent=1):
    async def background_indexer_loop():
        await asyncio.sleep(5)
        settings = Settings.load(app_data_dir=app_data_dir)
        print(f"[{datetime.now()}] Background RAG indexer started...")

        while True:
            try:
                books = scan_calibre_library(settings.calibre_library_path)
                indexing_dir = Path(settings.calibre_library_path)

                in_progress = list(indexing_dir.rglob("*.indexing"))
                if len(in_progress) >= max_concurrent:
                    print("Indexing is at max concurrency waiting")
                    await asyncio.sleep(1)
                    continue

                for book in books:
                    if not book.is_converted:
                        continue
                    if is_book_indexed(book):
                        continue

                    print(f"[DEBUG] Indexing book: {book.title}")
                    spawn_indexer_process(app_data_dir, book)
                    await asyncio.sleep(3)
                    break

                await asyncio.sleep(3)

            except Exception as e:
                print(f"[ERROR] Error in background task: {e}")
                await asyncio.sleep(3)

    return background_indexer_loop