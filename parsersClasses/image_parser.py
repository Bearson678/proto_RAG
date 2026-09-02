import os
from typing import List
from langchain_core.documents import Document
from parsersClasses.base_parser import BaseParser

class ImageParser(BaseParser):
    extensions = [".png", ".jpg", ".jpeg"]

    def __init__(self, ocr_client, chunk_size: int = 1000, chunk_overlap: int = 200):
        super().__init__(chunk_size, chunk_overlap)
        self.ocr_client = ocr_client  # whatever OCR/vision tool you plug in

    def parse_files(self, file_paths: List[str], **kwargs) -> List[Document]:
        documents = []
        for file_path in file_paths:
            if not os.path.isfile(file_path):
                print(f"Warning: File not found at {file_path}, skipping.")
                continue

            text = self.ocr_client.extract_text(file_path)  # your OCR call here
            source = self._normalize_source(file_path)
            documents.append(Document(page_content=text, metadata={"source": source}))

        return self.text_splitter.split_documents(documents)