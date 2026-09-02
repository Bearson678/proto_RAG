import os
from glob import glob
from typing import List, Optional, Dict
from langchain_core.documents import Document

from vectorDB.pinecone_db import PineconeDB
from vectorDB.embeder import EmbeddingGenerator
from parsersClasses.pdf_parser import PDFParser
from parsersClasses.csv_parser import CSVParser
from parsersClasses.image_parser import ImageParser
from parsersClasses.base_parser import BaseParser

from OcrClasses.tessaractOCRClient import TesseractOCRClient


class IngestionPipeline:
    def __init__(self):
        self.db = PineconeDB()
        self.embedder = EmbeddingGenerator(pc_client=self.db.get_pc())

        self._parser_instances: List[BaseParser] = [
            PDFParser(),
            CSVParser(),
            ImageParser(ocr_client=TesseractOCRClient()),
        ]

        self.parsers: Dict[str, BaseParser] = {
            ext: parser for parser in self._parser_instances for ext in parser.extensions
        }

    def route_files(self, file_paths: List[str]) -> Dict[str, List[str]]:
        """Groups file paths by extension, skipping types with no registered parser."""
        grouped: Dict[str, List[str]] = {ext: [] for ext in self.parsers}
        for path in file_paths:
            ext = os.path.splitext(path)[1].lower()
            if ext in self.parsers:
                grouped[ext].append(path)
            else:
                print(f"Warning: No parser registered for '{ext}' ({path}), skipping.")
        return grouped

    def parse_all(self, file_paths: List[str], csv_kwargs: Optional[dict] = None) -> List[Document]:
        csv_kwargs = csv_kwargs or {}
        grouped = self.route_files(file_paths)

        all_docs: List[Document] = []
        for ext, paths in grouped.items():
            if not paths:
                continue
            print(f"--- Parsing {len(paths)} '{ext}' file(s) ---")
            parser = self.parsers[ext]
            # Only CSVParser currently takes extra kwargs; keep the branch
            # explicit rather than forcing every parser to accept **kwargs.
            docs = parser.parse_files(paths, **csv_kwargs) if ext == ".csv" else parser.parse_files(paths)
            all_docs.extend(docs)

        return all_docs

    def embed_and_save(self, docs: List[Document], source_label: str):
        if not docs:
            print("No documents were extracted. Exiting pipeline.")
            return

        counts_by_source: Dict[str, int] = {}
        for doc in docs:
            src = doc.metadata.get("source", "unknown")
            counts_by_source[src] = counts_by_source.get(src, 0) + 1

        print("--- Chunk counts by source ---")
        for src, count in sorted(counts_by_source.items(), key=lambda x: -x[1]):
            print(f"  {src}: {count} chunks")

        print(f"--- Generating Embeddings for {len(docs)} Chunks ---")
        embeddings = self.embedder.generate_embeddings(docs)
        print("--- Saving Vectors to Pinecone ---")
        self.db.save_vectors(documents=docs, embeddings=embeddings, source_name=source_label)
        print("--- Pipeline Complete! ---")

    def process_directory(self, directory_path: str, source_label: str, csv_kwargs: Optional[dict] = None):
        """Ingests every supported file (mixed types OK) found directly in a
        directory. NOTE: loads every matched file fully into memory — route
        large CSVs through process_large_csv instead."""
        all_paths = []
        for ext in self.parsers:
            all_paths.extend(glob(os.path.join(directory_path, f"*{ext}")))
        all_paths = sorted(all_paths)

        if not all_paths:
            print(f"Warning: No supported files ({', '.join(self.parsers)}) found in {directory_path}.")
            return

        docs = self.parse_all(all_paths, csv_kwargs=csv_kwargs)
        self.embed_and_save(docs, source_label)

    def process_files(self, file_paths: List[str], source_label: str, csv_kwargs: Optional[dict] = None):
        """Ingests a specific, possibly mixed-type list of files — covers the one-by-one case."""
        docs = self.parse_all(file_paths, csv_kwargs=csv_kwargs)
        self.embed_and_save(docs, source_label)

    def process_large_csv(
        self,
        file_path: str,
        source_label: str,
        content_columns: Optional[List[str]] = None,
        id_column: Optional[str] = None,
        row_batch_size: int = 1000,
    ):
        """Streaming ingestion for CSVs too large to hold fully in memory:
        parse -> embed -> upsert per row window, instead of accumulating the
        whole file before touching Pinecone."""
        csv_parser: CSVParser = self.parsers[".csv"]
        index = self.db.get_pc().Index(self.db.get_index_name())
        source = csv_parser._normalize_source(file_path)

        # Clear old chunks for this file once, up front, rather than per
        # window. A crash mid-stream leaves partial data for this source —
        # re-running this method is the recovery path, since the upfront
        # delete makes that idempotent.
        self.db._delete_existing_for_source(index, source)

        total_chunks = 0
        for doc_batch in csv_parser.stream_files(
            [file_path], content_columns=content_columns, id_column=id_column, row_batch_size=row_batch_size
        ):
            if not doc_batch:
                continue
            embeddings = self.embedder.generate_embeddings(doc_batch)
            self.db.save_vectors(documents=doc_batch, embeddings=embeddings, source_name=source_label, skip_delete=True)
            total_chunks += len(doc_batch)
            print(f"Processed {total_chunks} chunks so far from {source}...")

        print(f"--- Finished streaming ingestion of {file_path}: {total_chunks} chunks ---")


if __name__ == "__main__":
    pipeline = IngestionPipeline()

    # Everything else — PDFs, images, smaller CSVs — fine to batch-process normally
    pipeline.process_files(
        file_paths=[
            "test_data/md_guidance-manufacturers_en.pdf",
            "test_data/CDS Project.pdf",
            "test_data/HASS Essay.pdf",
            "test_data/image.png",
        ],
        source_label="test_data",
    )

    # Whole directory in one call (mixed types OK) — avoid if it contains a
    # large CSV; use process_large_csv for that file instead
    # pipeline.process_directory(directory_path="test_data", source_label="test_data")