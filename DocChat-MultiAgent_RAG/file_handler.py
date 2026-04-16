import os
import hashlib
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import List

from docling.datamodel.settings import settings
from docling.document_converter import DocumentConverter
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langgraph import constants


class DocumentProcessor:
    def __init__(self):
        self.headers = [("#", "Header 1"), ("##", "Header 2")]
        self.cache_dir=Path(settings.CACHE_DIR)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    #     validates the total size of uploaded files
    def validate_files(self, files:List)->None:
        total_size=sum(os.path.getsize(f.name) for f in files)
        if total_size > constants.MAX_TOTAL_SIZE:
            raise ValueError(f"Total size exceeds {constants.MAX_TOTAL_SIZE // 1024 // 1024}MB limit")


    # process the file for caching
    # this one handles entire document processing pipeline

    # Validates the uploaded files
    # Generates a hash of each file's content to check if it has been processed before
    # If cached, loads the data from cache
    # If not cached, processes the file using _process_file() and stores the results in cache
    # Ensures that no duplicate chunks are stored across multiple files
    def process(self, files:List)->List:
        self.validate_files(files)
        all_chunks=[]
        seen_hashes=set()

        for file in files:
            try:
                # generating content-based hashing for caching
                with open(file.name, "rb") as f:
                    file_hash = self.generate_hash(f.read())
                cache_path=self.cache_dir
                if self.is_cache_valid(cache_path):
                    chunks=self.load_from_cache(cache_path)
                else:
                    chunks=self.process_file(file)
                    self.save_to_cache(chunks, cache_path)

                # Deduplicate chunks across files
                for chunk in chunks:
                    chunk_hash = self.generate_hash(chunk.page_content.encode())
                    if chunk_hash not in seen_hashes:
                        all_chunks.append(chunk)
                        seen_hashes.add(chunk_hash)

            except Exception as e:
                print(f"Failed to process {file.name}: {str(e)}")
                continue

        return all_chunks

    # processing logic with docling

    # Skips unsupported file types (only processes .pdf, .docx, .txt, and .md)
    # Uses Docling's DocumentConverter to convert the file to Markdown.
    # Splits the extracted Markdown text using MarkdownHeaderTextSplitter.
    def process_file(self, file)->List:
        if not file.name.endswith(('.pdf', '.docx', '.txt', '.md')):
            print(f"Skipping unsupported file type: {file.name}")
            return []
        converter=DocumentConverter()
        markdown=converter.convert(file.name).document.export_to_markdown()
        splitter=MarkdownHeaderTextSplitter(self.headers)
        return splitter.split_text(markdown)

    def generate_hash(self, content:bytes)->str:
        return hashlib.sha256(content).hexdigest()

    def save_to_cache(self, chunks:List, cache_path:Path):
        with open(cache_path, "wb")as f:
            pickle.dump({
                "timestamp": datetime.now().timestamp(),
                "chunks": chunks
            }, f)

    def load_from_cache(self, cache_path: Path) -> List:
        with open(cache_path, "rb") as f:
            data = pickle.load(f)
        return data["chunks"]


    # Compares the modification timestamp of the cached file against the CACHE_EXPIRE_DAYS setting.
    # If the file is older than the expiration threshold, it is considered invalid.
    def is_cache_valid(self, cache_path: Path) -> bool:
        if not cache_path.exists():
            return False

        cache_age = datetime.now() - datetime.fromtimestamp(cache_path.stat().st_mtime)
        return cache_age < timedelta(days=settings.CACHE_EXPIRE_DAYS)

