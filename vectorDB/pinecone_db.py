import os
from dotenv import load_dotenv
from pinecone import Pinecone,ServerlessSpec
from typing import List
from langchain_core.documents import Document
import hashlib
import math

INDEX_NAME = "protorag"
load_dotenv()
class PineconeDB:
    def __init__(self):
        
        #Load API key from .env file
        pinecone_api_key = os.getenv("PINECONE_API_KEY")
        if not pinecone_api_key:
            raise ValueError("PINECONE_API_KEY not set in .env file.")
        
        self.pc = Pinecone(api_key=pinecone_api_key)
        
        self.index_name = INDEX_NAME
        existing_indexes = [idx.name for idx in self.pc.list_indexes()]
        if self.index_name not in existing_indexes:
            self.pc.create_index(
                name=self.index_name, 
                dimension=1024, 
                metric="cosine", 
                serverless=ServerlessSpec(cloud='aws', region='us-east-1')
            )
            
    @staticmethod
    def _make_vector_id(doc: Document, fallback_source: str, fallback_index: int) -> str:
        source = doc.metadata.get("source", fallback_source)
        position = doc.metadata.get("page", doc.metadata.get("row", fallback_index))
        content_hash = hashlib.sha1(doc.page_content.encode("utf-8")).hexdigest()[:8]
        return f"{source}_{position}_{content_hash}"

    @staticmethod
    def _sanitize_metadata(metadata: dict) -> dict:
        """Pinecone rejects None/NaN metadata values — strip them (common with
        empty CSV cells)."""
        clean = {}
        for k, v in metadata.items():
            if v is None:
                continue
            if isinstance(v, float) and math.isnan(v):
                continue
            clean[k] = v
        return clean

    def _delete_existing_for_source(self, index, source: str):
        """Removes any previously ingested chunks for this source before
        re-upserting, so edited/re-ingested files don't leave orphaned vectors."""
        try:
            index.delete(filter={"source": {"$eq": source}})
        except Exception as e:
            print(f"Warning: could not clear existing vectors for '{source}': {e}")

    def save_vectors(self, documents, embeddings, source_name="pdf_upload", skip_delete=False, upsert_batch_size=100):
        index = self.pc.Index(self.index_name)

        if not skip_delete:
            sources = {doc.metadata.get("source", source_name) for doc in documents}
            for source in sources:
                self._delete_existing_for_source(index, source)

        vectors_to_upsert = []
        for i, (doc, vector) in enumerate(zip(documents, embeddings)):
            vector_id = self._make_vector_id(doc, fallback_source=source_name, fallback_index=i)
            chunk_metadata = self._sanitize_metadata({
                **doc.metadata,
                "text": doc.page_content,
                "chunk_index": i
            })
            vectors_to_upsert.append((vector_id, vector, chunk_metadata))

        for i in range(0, len(vectors_to_upsert), upsert_batch_size):
            batch = vectors_to_upsert[i:i + upsert_batch_size]
            index.upsert(vectors=batch)

        if vectors_to_upsert:
            print(f"Successfully uploaded {len(vectors_to_upsert)} vectors to Pinecone.")
            
    def get_index_name(self):
        return self.index_name
    
    def get_pc(self):
        return self.pc