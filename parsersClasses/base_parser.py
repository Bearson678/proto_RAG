import os
from abc import ABC, abstractmethod
from glob import glob
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class BaseParser(ABC):
    """Common interface for all ingestion parsers (PDF, CSV, Image, ...).

    Subclasses declare `extensions` and implement `parse_files`.
    `parse_directory` and source normalization are shared here so every
    new parser gets consistent, deterministic behavior for free.
    """

    extensions: List[str] = []  # e.g. [".pdf"], [".csv"], [".png", ".jpg"]

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    @abstractmethod
    def parse_files(self, file_paths: List[str], **kwargs) -> List[Document]:
        """Parse a specific list of files into chunked Documents."""
        raise NotImplementedError

    def parse_directory(self, directory_path: str, **kwargs) -> List[Document]:
        """Shared directory scan — subclasses shouldn't need to override this."""
        matched_files = []
        for ext in self.extensions:
            matched_files.extend(glob(os.path.join(directory_path, f"*{ext}")))
        matched_files = sorted(matched_files)

        if not matched_files:
            print(f"Warning: No {'/'.join(self.extensions)} files found in {directory_path}.")
            return []
        return self.parse_files(matched_files, **kwargs)

    @staticmethod
    def _normalize_source(file_path: str) -> str:
        """Stable identifier for a file, regardless of how its path was passed in."""
        return os.path.basename(file_path)