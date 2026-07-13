# crawler.py
import os
import json
import asyncio
from typing import List, Dict, Any
from pydantic import BaseModel, Field
from crawl4ai import AsyncWebCrawler, CacheMode
from crawl4ai.async_configs import CrawlerRunConfig
from crawl4ai.extraction_strategy import LLMExtractionStrategy, create_llm_config

# Schema rules passed directly to Groq
class CreditCardSchema(BaseModel):
    card_name: str = Field(..., description="Official proper name of the card. Return 'SKIP' if not a card profile.")
    category: str = Field(..., description="Travel Rewards, Cash Back, Balance Transfer, Student, etc.")
    features: str = Field(..., description="Core reward multipliers or key earning structures.")
    benefits: str = Field(..., description="Sign-up offers, statement credits, or perks.")
    rates_fees: str = Field(..., description="Annual fee details and APR specifications.")
    product_url: str = Field(..., description="Hyperlink string if available, else return 'Link in Main Catalog'.")

async def fetch_raw_markdown(url: str) -> str:
    """Uses Crawl4AI to parse layout content strings down cleanly."""
    async with AsyncWebCrawler(verbose=False) as crawler:
        result = await crawler.arun(
            url=url, 
            config=CrawlerRunConfig(cache_mode=CacheMode.BYPASS, excluded_tags=["nav", "footer"])
        )
    if not result.success:
        raise RuntimeError(f"Crawl operations failed: {result.error_message}")
    return result.markdown.fit_markdown or result.markdown.raw_markdown

async def extract_card_properties_from_chunk(chunk_text: str, source_label: str, chunk_idx: int) -> List[Dict[str, Any]]:
    """Runs a single preprocessed metadata slice securely through the Groq extraction layer."""
    groq_key = os.environ.get("GROQ_API_KEY")
    if not groq_key:
        raise ValueError("Error: Environment variable GROQ_API_KEY is missing.")
        
    llm_config = create_llm_config(provider="groq/llama-3.3-70b-versatile", api_token=groq_key)
    strategy = LLMExtractionStrategy(
        llm_config=llm_config, 
        extraction_type="schema",
        schema=CreditCardSchema.model_json_schema(), 
        input_format="fit_markdown",
        instruction="Extract all listed credit card offers into structured JSON matching the schema.", 
        verbose=False
    )
    
    async with AsyncWebCrawler(verbose=False) as crawler:
        safe_virtual_url = f"raw://data_pipeline/{source_label}/chunk_{chunk_idx}/{chunk_text}"
        result = await crawler.arun(
            url=safe_virtual_url, 
            config=CrawlerRunConfig(extraction_strategy=strategy, cache_mode=CacheMode.BYPASS)
        )
        
    if result.success and result.extracted_content:
        try:
            parsed = json.loads(result.extracted_content)
            return parsed if isinstance(parsed, list) else [parsed]
        except json.JSONDecodeError:
            return []
    return []