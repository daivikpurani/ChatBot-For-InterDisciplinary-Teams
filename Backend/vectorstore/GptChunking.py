import os
import uuid
import glob
import asyncio
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import List, Dict, Any
import numpy as np
from langchain_ollama import ChatOllama
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from pydantic import BaseModel
from typing import Optional
from langchain.chains import create_extraction_chain_pydantic
from pinecone import Pinecone, ServerlessSpec
from tqdm import tqdm
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
from rich.panel import Panel
from rich import print as rprint
import datetime

# Initialize rich console
console = Console()

# Load environment variables
load_dotenv()

# === CONFIG ===
LOCAL_MODEL_NAME = "llama3.2"
EMBEDDING_MODEL_NAME = "text-embedding-ada-002"
PINECONE_INDEX_NAME = "travel-data-agentic-chunks"
PDF_FOLDER = "/Users/daivikpurani/Desktop/ACAD/Thesis/code/Backend/backupPDF"
TEXT_FOLDER = "/Users/daivikpurani/Desktop/ACAD/Thesis/code/Backend/scrapeddata"

# Chunking configuration
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
BATCH_SIZE = 8  # Reduced batch size to lower memory usage
MAX_WORKERS = 2  # Reduced number of workers
EMBEDDING_DIMENSION = 1536  # Dimension for text-embedding-ada-002
COOLDOWN_SECONDS = 1  # Seconds to wait between batches
SIMILARITY_THRESHOLD = 0.98  # Increased threshold to reduce false positives

# === SETUP ===
console.print(Panel.fit("Initializing Models and Services", style="bold blue"))

with console.status("[bold green]Initializing LLM...") as status:
    local_llm = ChatOllama(model=LOCAL_MODEL_NAME)
    console.print("[green]✓[/green] LLM initialized")

with console.status("[bold green]Initializing Embedding Model...") as status:
    try:
        embedding_model = OpenAIEmbeddings(
            model=EMBEDDING_MODEL_NAME,
            openai_api_key="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"
        )
        console.print("[green]✓[/green] OpenAI embedding model initialized")
        # Verify the embedding dimension
        test_embedding = embedding_model.embed_query("test")
        actual_dimension = len(test_embedding)
        if actual_dimension != EMBEDDING_DIMENSION:
            console.print(f"[yellow]Warning: Model produced {actual_dimension} dimensions, expected {EMBEDDING_DIMENSION}[/yellow]")
            EMBEDDING_DIMENSION = actual_dimension
    except Exception as e:
        console.print(f"[red]Error initializing OpenAI embedding model: {e}")
        raise e  # We don't have a fallback for OpenAI, so we'll raise the error

with console.status("[bold green]Initializing Pinecone...") as status:
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    
    if 'travel-data-agentic-chunks' not in pc.list_indexes().names():
        console.print("[yellow]Creating new Pinecone index...[/yellow]")
        pc.create_index(
            name='travel-data-agentic-chunks',
            dimension=EMBEDDING_DIMENSION,  # Use actual model dimension
            metric='euclidean',
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'
            )
        )
        console.print("[green]✓[/green] New Pinecone index created")
    else:
        console.print("[green]✓[/green] Existing Pinecone index found")
    
    pinecone_index = pc.Index(PINECONE_INDEX_NAME)

class OptimizedAgenticChunker:
    def __init__(self):
        self.chunks = {}
        self.llm = ChatOllama(model=LOCAL_MODEL_NAME)
        self.id_truncate_limit = 5
        self.print_logging = True
        self.batch_size = BATCH_SIZE
        self.propositions_buffer = []

    async def process_batch(self, propositions: List[str]):
        """Process a batch of propositions together"""
        if not propositions:
            return

        # Get chunk assignments for all propositions in batch
        chunk_assignments = await self._batch_find_relevant_chunks(propositions)
        
        # Process each proposition based on its assignment
        for prop, chunk_id in zip(propositions, chunk_assignments):
            if chunk_id:
                self.chunks[chunk_id]['propositions'].append(prop)
            else:
                self._create_new_chunk(prop)

        # Update summaries and titles in batch
        await self._batch_update_chunk_metadata()

    async def _batch_find_relevant_chunks(self, propositions: List[str]) -> List[str]:
        """Find relevant chunks for a batch of propositions"""
        current_chunk_outline = self.get_chunk_outline()
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """
            For each proposition, determine if it should belong to any existing chunk.
            Return a comma-separated list of chunk IDs or "No chunks" for each proposition.
            Each response should be either a valid chunk ID (5 characters) or "No chunks".
            Do not include any other text in your response.

            Example valid responses:
            "2n4l3,No chunks,93833"
            "No chunks,2n4l3,No chunks"
            "93833,2n4l3,No chunks"

            Invalid responses (DO NOT USE):
            "This should go in chunk 2n4l3"
            "The first proposition belongs to chunk 93833"
            "No matching chunks found for this"
            """),
            ("user", "Current Chunks:\n{current_chunk_outline}\n\nPropositions:\n{propositions}")
        ])

        chain = prompt | self.llm | StrOutputParser()
        response = await chain.ainvoke({
            "current_chunk_outline": current_chunk_outline,
            "propositions": "\n".join(propositions)
        })

        # Clean and validate the response
        chunk_ids = []
        for id_str in response.split(","):
            id_str = id_str.strip()
            # Validate chunk ID format
            if id_str == "No chunks":
                chunk_ids.append(None)
            elif len(id_str) == self.id_truncate_limit and id_str.isalnum():
                # Only use the ID if it exists in our chunks
                if id_str in self.chunks:
                    chunk_ids.append(id_str)
                else:
                    chunk_ids.append(None)
            else:
                chunk_ids.append(None)

        # Ensure we have the same number of IDs as propositions
        if len(chunk_ids) != len(propositions):
            console.print(f"[yellow]Warning: Mismatch in chunk assignments. Expected {len(propositions)}, got {len(chunk_ids)}[/yellow]")
            # Pad with None if we got fewer IDs
            chunk_ids.extend([None] * (len(propositions) - len(chunk_ids)))
            # Truncate if we got more IDs
            chunk_ids = chunk_ids[:len(propositions)]

        return chunk_ids

    async def _batch_update_chunk_metadata(self):
        """Update summaries and titles for all chunks in batch"""
        for chunk_id, chunk in self.chunks.items():
            chunk['summary'] = await self._get_updated_chunk_summary(chunk)
            chunk['title'] = await self._get_updated_chunk_title(chunk)

    async def _get_updated_chunk_summary(self, chunk):
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Generate a brief summary for this chunk of propositions."),
            ("user", "Propositions:\n{propositions}")
        ])
        chain = prompt | self.llm | StrOutputParser()
        return await chain.ainvoke({"propositions": "\n".join(chunk['propositions'])})

    async def _get_updated_chunk_title(self, chunk):
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Generate a brief title for this chunk."),
            ("user", "Summary: {summary}")
        ])
        chain = prompt | self.llm | StrOutputParser()
        return await chain.ainvoke({"summary": chunk['summary']})

    def _create_new_chunk(self, proposition):
        new_chunk_id = str(uuid.uuid4())[:self.id_truncate_limit]
        self.chunks[new_chunk_id] = {
            'chunk_id': new_chunk_id,
            'propositions': [proposition],
            'summary': "",  # Will be updated in batch
            'title': ""     # Will be updated in batch
        }

    def get_chunk_outline(self):
        return "\n".join([
            f"Chunk ID: {chunk_id}\nTitle: {chunk['title']}\nSummary: {chunk['summary']}\n"
            for chunk_id, chunk in self.chunks.items()
        ])

    def get_chunks(self, get_type='dict'):
        if get_type == 'dict':
            return self.chunks
        if get_type == 'list_of_strings':
            return [" ".join(chunk['propositions']) for chunk in self.chunks.values()]

async def load_and_split_text_files_async(text_folder):
    """Asynchronously load and split text files"""
    console.print(Panel.fit("Loading Text Files", style="bold blue"))
    
    documents = []
    text_files = glob.glob(os.path.join(text_folder, "*.txt"))
    
    if not text_files:
        console.print("[red]Warning: No text files found in the specified directory![/red]")
        return documents

    async def load_file(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                return Document(page_content=content, metadata={"source": file_path})
        except Exception as e:
            console.print(f"[red]Error loading {file_path}: {str(e)}[/red]")
            return None

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
    ) as progress:
        task = progress.add_task("[cyan]Loading files...", total=len(text_files))
        
        # Load files in parallel
        tasks = [load_file(file_path) for file_path in text_files]
        loaded_docs = await asyncio.gather(*tasks)
        documents.extend([doc for doc in loaded_docs if doc is not None])
        progress.update(task, completed=len(text_files))
    
    console.print(f"[green]✓[/green] Loaded {len(documents)} total text documents")
    return documents

async def batch_embed_documents(documents, batch_size=BATCH_SIZE):
    """Generate embeddings in batches"""
    console.print(Panel.fit("Generating Embeddings", style="bold blue"))
    
    embeddings = []
    total_batches = (len(documents) + batch_size - 1) // batch_size
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
    ) as progress:
        task = progress.add_task("[cyan]Generating embeddings...", total=len(documents))
        
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            batch_embeddings = await asyncio.gather(*[
                embedding_model.aembed_query(doc.page_content if hasattr(doc, 'page_content') else doc)
                for doc in batch
            ])
            embeddings.extend(batch_embeddings)
            progress.update(task, advance=len(batch))
    
    console.print(f"[green]✓[/green] Generated {len(embeddings)} embeddings")
    return embeddings

async def check_chunk_exists(index, chunk_text):
    """Check if a chunk already exists in Pinecone"""
    try:
        # Generate embedding for the chunk
        embedding = await embedding_model.aembed_query(chunk_text)
        
        # Query Pinecone with the embedding
        results = await index.query(
            vector=embedding,
            top_k=1,
            include_metadata=True
        )
        
        # Check if any results are very similar
        if results.matches:
            best_match = results.matches[0]
            if best_match.score > SIMILARITY_THRESHOLD:
                return True, best_match.id
        return False, None
    except Exception as e:
        console.print(f"[red]Error checking chunk existence: {str(e)}[/red]")
        return False, None

async def stream_and_upload_chunks(chunks, index):
    """Stream chunks and upload only new ones to Pinecone with cooldown periods"""
    console.print(Panel.fit("Streaming and Uploading Chunks", style="bold blue"))
    
    total_chunks = len(chunks)
    uploaded_count = 0
    skipped_count = 0
    error_count = 0
    processed_count = 0
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
    ) as progress:
        task = progress.add_task("[cyan]Processing chunks...", total=total_chunks)
        
        for i in range(0, total_chunks, BATCH_SIZE):
            batch = chunks[i:i + BATCH_SIZE]
            
            # Process batch
            for chunk in batch:
                processed_count += 1
                chunk_text = chunk.page_content if hasattr(chunk, 'page_content') else chunk
                
                try:
                    # Check if chunk exists
                    exists, existing_id = await check_chunk_exists(index, chunk_text)
                    
                    if exists:
                        console.print(f"[yellow]Chunk #{processed_count}/{total_chunks} already exists (ID: {existing_id}). Skipping...[/yellow]")
                        skipped_count += 1
                    else:
                        # Generate embedding for new chunk
                        embedding = await embedding_model.aembed_query(chunk_text)
                        
                        # Prepare vector for upload
                        vector = {
                            'id': f'vec_{uuid.uuid4().hex[:8]}',
                            'values': embedding,
                            'metadata': {
                                'text': chunk_text,
                                'source': 'local',
                                'timestamp': str(datetime.datetime.now()),
                                'chunk_number': processed_count
                            }
                        }
                        
                        # Upload single vector
                        await index.upsert(vectors=[vector])
                        uploaded_count += 1
                        console.print(f"[green]✓ Chunk #{processed_count}/{total_chunks} uploaded successfully[/green]")
                        console.print(f"[blue]Running totals - Uploaded: {uploaded_count}, Skipped: {skipped_count}, Errors: {error_count}[/blue]")
                
                except Exception as e:
                    error_count += 1
                    console.print(f"[red]Error processing Chunk #{processed_count}/{total_chunks}: {str(e)}[/red]")
                
                progress.update(task, advance=1)
            
            # Cooldown period between batches
            if i + BATCH_SIZE < total_chunks:
                console.print(f"[yellow]Cooling down for {COOLDOWN_SECONDS} seconds...[/yellow]")
                await asyncio.sleep(COOLDOWN_SECONDS)
    
    console.print(Panel.fit("Upload Summary", style="bold green"))
    console.print(f"Total chunks processed: {total_chunks}")
    console.print(f"Chunks uploaded: {uploaded_count}")
    console.print(f"Chunks skipped (already exist): {skipped_count}")
    console.print(f"Errors encountered: {error_count}")
    console.print(f"Success rate: {((uploaded_count + skipped_count) / total_chunks * 100):.1f}%")

async def main():
    console.print(Panel.fit("AGENTIC RAG PIPELINE", style="bold blue"))
    
    # Load text files asynchronously
    text_documents = await load_and_split_text_files_async(TEXT_FOLDER)
    
    if not text_documents:
        console.print("[red]No documents found. Exiting...[/red]")
        return
    
    # Split documents into chunks
    console.print(Panel.fit("Splitting Documents", style="bold blue"))
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    all_chunks = splitter.split_documents(text_documents)
    console.print(f"[green]✓[/green] Split into {len(all_chunks)} total chunks")
    
    if not all_chunks:
        console.print("[red]No chunks generated. Exiting...[/red]")
        return

    # Process with optimized agentic chunker
    console.print(Panel.fit("Processing with Agentic Chunker", style="bold blue"))
    chunker = OptimizedAgenticChunker()
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
    ) as progress:
        task = progress.add_task("[cyan]Processing chunks...", total=len(all_chunks))
        
        for i in range(0, len(all_chunks), BATCH_SIZE):
            batch = all_chunks[i:i + BATCH_SIZE]
            propositions = [chunk.page_content for chunk in batch]
            await chunker.process_batch(propositions)
            progress.update(task, advance=len(batch))
            
            # Cooldown period between batches
            if i + BATCH_SIZE < len(all_chunks):
                await asyncio.sleep(COOLDOWN_SECONDS)
    
    # Get processed chunks
    agentic_chunks = chunker.get_chunks(get_type='list_of_strings')
    console.print(f"[green]✓[/green] Generated {len(agentic_chunks)} agentic chunks")
    
    # Stream and upload chunks to Pinecone
    await stream_and_upload_chunks(agentic_chunks, pinecone_index)
    
    console.print(Panel.fit("PIPELINE COMPLETED", style="bold green"))
    console.print(f"Total documents processed: {len(text_documents)}")
    console.print(f"Total chunks generated: {len(all_chunks)}")
    console.print(f"Total agentic chunks: {len(agentic_chunks)}")

async def test_embeddings():
    """Test function to verify OpenAI embeddings are working"""
    console.print(Panel.fit("Testing OpenAI Embeddings", style="bold blue"))
    
    test_texts = [
        "This is a test sentence for embedding.",
        "Another test sentence to verify the model.",
        "Testing the OpenAI text-embedding-ada-002 model."
    ]
    
    try:
        # Test single embedding
        console.print("[cyan]Testing single text embedding...[/cyan]")
        single_embedding = await embedding_model.aembed_query(test_texts[0])
        console.print(f"[green]✓[/green] Single embedding successful. Dimension: {len(single_embedding)}")
        
        # Test batch embedding
        console.print("[cyan]Testing batch embedding...[/cyan]")
        batch_embeddings = await asyncio.gather(*[
            embedding_model.aembed_query(text)
            for text in test_texts
        ])
        console.print(f"[green]✓[/green] Batch embedding successful. Generated {len(batch_embeddings)} embeddings")
        
        # Verify dimensions
        dimensions = [len(emb) for emb in batch_embeddings]
        if all(dim == EMBEDDING_DIMENSION for dim in dimensions):
            console.print(f"[green]✓[/green] All embeddings have correct dimension: {EMBEDDING_DIMENSION}")
        else:
            console.print(f"[red]Error: Inconsistent embedding dimensions: {dimensions}[/red]")
        
        return True
    except Exception as e:
        console.print(f"[red]Error during embedding test: {str(e)}[/red]")
        return False

if __name__ == "__main__":
    # Run the test first
    if asyncio.run(test_embeddings()):
        console.print("[green]Embedding test successful! Proceeding with main pipeline...[/green]")
        asyncio.run(main())
    else:
        console.print("[red]Embedding test failed! Please check your configuration.[/red]")
    
    
