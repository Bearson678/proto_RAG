import os
from langchain_community.document_loaders import PyPDFLoader
from typing import List
from langchain_core.documents import Document
from parsersClasses.base_parser import BaseParser


class PDFParser(BaseParser):
    extensions = [".pdf"]

    def parse_files(self, file_paths: List[str], **kwargs) -> List[Document]:
        documents = []
        for file_path in file_paths:
            if not os.path.isfile(file_path):
                print(f"Warning: File not found at {file_path}, skipping.")
                continue
            loader = PyPDFLoader(file_path)
            loaded = loader.load()

            source = self._normalize_source(file_path)
            for doc in loaded:
                doc.metadata["source"] = source

            documents.extend(loaded)
        return self.text_splitter.split_documents(documents)