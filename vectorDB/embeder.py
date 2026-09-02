import os
from typing import List
from langchain_core.documents import Document
from pinecone import Pinecone

class EmbeddingGenerator:
    # llama-text-embed-v2's documented max_batch_size is 96 inputs per request
    BATCH_SIZE = 96

    def __init__(self, pc_client: Pinecone):
        self.pc = pc_client

    def generate_embeddings(self, documents: List[Document]) -> List[List[float]]:
        if not documents:
            return []

        texts = [doc.page_content for doc in documents]
        all_embeddings: List[List[float]] = []

        for i in range(0, len(texts), self.BATCH_SIZE):
            batch = texts[i:i + self.BATCH_SIZE]
            response = self.pc.inference.embed(
                model="llama-text-embed-v2",
                inputs=batch,
                parameters={"input_type": "passage", "truncate": "END"}
            )
            # Safely parse out the raw float vectors from Pinecone API payload
            batch_embeddings = [item["values"] for item in response]
            all_embeddings.extend(batch_embeddings)
            print(f"Embedded batch {i // self.BATCH_SIZE + 1} ({len(batch)} chunks)")

        return all_embeddings