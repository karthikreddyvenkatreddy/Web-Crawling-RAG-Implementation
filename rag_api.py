# rag_api.py
import os
import groq
from typing import List
from index import CreditCardVectorStoreIndex

class RAGPipeline:
    def __init__(self, vector_store: CreditCardVectorStoreIndex):
        self.vector_store = vector_store
        self.groq_client = None
        
    def _get_client(self):
        if self.groq_client is None:
            groq_key = os.environ.get("GROQ_API_KEY")
            if not groq_key:
                raise ValueError("Environment variable GROQ_API_KEY is missing.")
            self.groq_client = groq.Groq(api_key=groq_key)
        return self.groq_client

    def answer_query(self, user_query: str) -> str:
        """Retrieves top local context blocks and returns a verified LLM completion."""
        matched_excerpts = self.vector_store.retrieve_top_k(user_query, k=3)
        context_buffer = "\n\n---\n\n".join(matched_excerpts)

        system_prompt = (
            "You are an expert Bank of America credit product assistant. Your goal is to answer user queries accurately. "
            "Base your answer ONLY on the provided verified credit card contexts excerpts below. "
            "Always explicitly cite the card names and provide rates, fee constraints, or URLs when available. "
            "If the document context doesn't contain the answer, politely state that the data is not present in the catalog sources."
        )

        rag_user_prompt = f"""
[CONTEXT EXCERPTS]:
{context_buffer}

[USER QUESTION]:
{user_query}

Please formulate an accurate, comprehensive answer citing relevant details from the context:
"""
        try:
            client = self._get_client()
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile", # Uses the large model for smart RAG synthesis
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": rag_user_prompt}
                ],
                temperature=0.1
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error executing generation inference pipeline: {str(e)}"