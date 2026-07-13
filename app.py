# app.py
import json
import os
import asyncio
import gradio as gr

from crawler import fetch_raw_markdown, extract_card_properties_from_chunk
from preprocess import clean_and_normalize_text, token_aware_chunk_splitter, sanitize_and_deduplicate_records
from index import CreditCardVectorStoreIndex
from rag_api import RAGPipeline

URL = "https://www.bankofamerica.com/credit-cards/"
JSON_BACKUP = "bofa_cards_raw_backup.json"

db_path = os.path.abspath("./local_chroma_db")
vector_store = CreditCardVectorStoreIndex(db_directory=db_path)
rag_system = RAGPipeline(vector_store)

async def initialize_and_build_pipeline():
    """Prioritizes local JSON backups and Chroma files to bypass live web crawling entirely."""
    # Scenario 1: Chroma folder exists and is already loaded
    if vector_store.is_already_indexed():
        print("Local Chroma DB files detected! System loaded instantly.")
        return

    # Scenario 2: Chroma was deleted, but the local JSON backup file exists
    if os.path.exists(JSON_BACKUP):
        print(f"Found local JSON backup ({JSON_BACKUP})! Seeding Chroma database directly...")
        with open(JSON_BACKUP, "r", encoding="utf-8") as file_in:
            saved_data = json.load(file_in)
        
        # Build the Chroma database folder instantly from the file data
        vector_store.build_index_from_json(saved_data)
        print("Chroma DB successfully rebuilt from local file cache. Startup complete!")
        return

    # Scenario 3: Neither exist (First time setup scenario)
    print("Warning: No local database cache or JSON backup found. Commencing pipeline crawl...")
    raw_markdown = await fetch_raw_markdown(URL)
    cleaned_page_text = clean_and_normalize_text(raw_markdown)
    metadata_chunks = list(token_aware_chunk_splitter(cleaned_page_text, "credit-card-landing-page"))
    
    all_raw_extracted = []
    for chunk_obj in metadata_chunks:
        text, source, idx = chunk_obj["text"], chunk_obj["source"], chunk_obj["chunk_index"]
        print(f"Scraping chunk {idx + 1}/{len(metadata_chunks)}...")
        
        extracted_cards = await extract_card_properties_from_chunk(text, source, idx)
        for card in extracted_cards:
            card["data_source"] = source
            all_raw_extracted.append(card)
            
        if idx < len(metadata_chunks) - 1:
            print("Sleeping 25 seconds to protect Groq TPM limit...")
            await asyncio.sleep(25)
            
    final_cleaned_data = sanitize_and_deduplicate_records(all_raw_extracted)
    
    print(f"Saving structured datasets directly to local backup: {JSON_BACKUP}...")
    with open(JSON_BACKUP, "w", encoding="utf-8") as file_out:
        json.dump(final_cleaned_data, file_out, indent=4)
        
    vector_store.build_index_from_json(final_cleaned_data)
    print("\nRAG Backend Core Engine Initialized Successfully!")

# Define UI Panel
app_interface = gr.Interface(
    fn=rag_system.answer_query,
    inputs=gr.Textbox(lines=2, placeholder="Ask anything, e.g., 'Which student cards offer 0% APR on balance transfers?'"),
    outputs=gr.Markdown(),
    title="Bank of America Credit Products - Smart Local Chroma RAG",
    description="Interactive query assistant operating completely locally using Chroma DB.",
    theme="soft"
)

if __name__ == "__main__":
    print("Launching local pipeline framework setup...")
    
    # Run the async database check/build inside the existing loop before launching the UI
    loop = asyncio.get_event_loop()
    loop.run_until_complete(initialize_and_build_pipeline())
    
    # Launch Gradio server application
    app_interface.launch()
