import os
import pandas as pd
from typing import List, Optional
from langchain_core.documents import Document
from parsersClasses.base_parser import BaseParser


class CSVParser(BaseParser):
    extensions = [".csv"]

    def _row_to_document(self, row, row_position, source, content_columns, id_column):
        if content_columns:
            missing = [c for c in content_columns if c not in row.index]
            if missing:
                raise ValueError(f"content_columns {missing} not found in CSV columns: {list(row.index)}")
            text_parts = [f"{col}: {row[col]}" for col in content_columns]
        else:
            text_parts = [f"{col}: {row[col]}" for col in row.index]
        content = "\n".join(text_parts)

        metadata = {col: row[col] for col in row.index}
        metadata["source"] = source
        metadata["row"] = row[id_column] if id_column else row_position

        return Document(page_content=content, metadata=metadata)

    def parse_files(
        self,
        file_paths: List[str],
        content_columns: Optional[List[str]] = None,
        id_column: Optional[str] = None,
        **kwargs,
    ) -> List[Document]:
        documents = []
        for file_path in file_paths:
            if not os.path.isfile(file_path):
                print(f"Warning: File not found at {file_path}, skipping.")
                continue

            source = self._normalize_source(file_path)
            df = pd.read_csv(file_path)

            if id_column and id_column not in df.columns:
                raise ValueError(f"id_column '{id_column}' not found in {file_path}'s columns: {list(df.columns)}")

            for row_position, row in df.iterrows():
                documents.append(self._row_to_document(row, row_position, source, content_columns, id_column))

        return self.text_splitter.split_documents(documents)
    
    def stream_files(self, file_paths, content_columns=None, id_column=None, row_batch_size=1000):
        """Yields batches of chunked Documents, reading each CSV in row windows
        instead of loading the whole file into memory at once."""
        for file_path in file_paths:
            if not os.path.isfile(file_path):
                print(f"Warning: File not found at {file_path}, skipping.")
                continue

            source = self._normalize_source(file_path)

            if id_column:
                header_cols = pd.read_csv(file_path, nrows=0).columns
                if id_column not in header_cols:
                    raise ValueError(f"id_column '{id_column}' not found in {file_path}'s columns: {list(header_cols)}")

            row_position = 0
            for chunk_df in pd.read_csv(file_path, chunksize=row_batch_size):
                batch_docs = []
                for _, row in chunk_df.iterrows():
                    batch_docs.append(self._row_to_document(row, row_position, source, content_columns, id_column))
                    row_position += 1
                yield self.text_splitter.split_documents(batch_docs)    