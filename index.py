# index.py
import os
import re
from typing import List, Dict, Any
import chromadb

class CreditCardVectorStoreIndex:
    def __init__(self, db_directory: str = "./local_chroma_db"):
        self.db_directory = db_directory
        # Initialize a 100% local persistent disk client 
        self.chroma_client = chromadb.PersistentClient(path=self.db_directory)
        # Using Chroma's default built-in lightweight keyword/vector parsing strategy
        self.collection = self.chroma_client.get_or_create_collection(name="bofa_local_cards")

    def is_already_indexed(self) -> bool:
        """Checks if the local Chroma folder already has data."""
        return self.collection.count() > 0

    def build_index_from_json(self, card_records: List[Dict[str, Any]]):
        """Saves the card metrics directly into the local Chroma disk folder."""
        if self.is_already_indexed():
            print("-> Found existing data cached inside local ChromaDB storage. Skipping scrape.")
            return

        print(f"-> Indexing {len(card_records)} cards into local Chroma database...")
        
        documents = []
        metadatas = []
        ids = []

        for idx, card in enumerate(card_records):
            semantic_text_block = (
                f"Card Name: {card['card_name']}\n"
                f"Category: {card['category']}\n"
                f"Features: {card['features']}\n"
                f"Benefits: {card['benefits']}\n"
                f"Rates & Fees: {card['rates_fees']}\n"
                f"Apply URL: {card['product_url']}"
            )
            documents.append(semantic_text_block)
            ids.append(f"card_{idx}")
            metadatas.append({"card_name": card['card_name']})

        # Save to local disk cache structure
        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        print("-> Vector dataset successfully saved to local Chroma DB files.")

    def retrieve_top_k(self, query: str, k: int = 3) -> List[str]:
        """Queries the local Chroma collection directly for match excerpts."""
        if self.collection.count() == 0:
            return ["No data found in local Chroma DB index."]

        results = self.collection.query(
            query_texts=[query],
            n_results=min(k, self.collection.count())
        )
        
        if results and 'documents' in results and results['documents']:
            return results['documents'][0]
        return ["No close context matches found."]