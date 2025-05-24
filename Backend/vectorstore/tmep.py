import os
import uuid
import glob
from langchain_ollama import ChatOllama
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
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

from rich import print

# Load environment variables
load_dotenv()

# === CONFIG ===
LOCAL_MODEL_NAME = "llama3.2"  # or "llama3.3"
EMBEDDING_MODEL_NAME = "sentence-transformers/sentence-t5-large"  # Best for RAG with academic content
PINECONE_INDEX_NAME = "agentic-chunks"
PDF_FOLDER = "/Users/daivikpurani/Desktop/ACAD/Thesis/code/Backend/backupPDF"  # Absolute path to PDF folder

# === SETUP ===
local_llm = ChatOllama(model=LOCAL_MODEL_NAME)
try:
    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cpu'},  # Force CPU usage to avoid GPU issues
        encode_kwargs={'normalize_embeddings': True}
    )
except Exception as e:
    print(f"Error initializing embedding model: {e}")
    print("Falling back to default embedding model...")
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

# Initialize Pinecone
PINECONE_API_KEY=os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key=PINECONE_API_KEY)

# Create or connect to Pinecone index
if 'agentic-chunks' not in pc.list_indexes().names():
        pc.create_index(
            name='agentic-chunks', 
            dimension=1536,  # Free tier maximum dimension
            metric='euclidean',
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'
        )
)

pinecone_index = pc.Index(PINECONE_INDEX_NAME)

# === BUILD RAG CHAIN ===
def build_rag_chain(retriever):
    prompt_template = """
You are a knowledgeable and helpful AI assistant.
You must answer the user's question using only the information provided in the context below.

Instructions:
- Use only the provided context to answer the question.
- If the answer cannot be found in the context, say: "I'm sorry, I couldn't find the answer based on the provided information."
- Do not hallucinate or make up any information.
- Be concise and accurate.

Context:
{context}

Question:
{question}

Answer:
"""
    prompt = ChatPromptTemplate.from_template(prompt_template)

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | local_llm
        | StrOutputParser()
    )
    return chain

# === AGENTIC CHUNKER ===
class AgenticChunker:
    def __init__(self):
        self.chunks = {}
        self.llm = ChatOllama(model=LOCAL_MODEL_NAME)
        self.id_truncate_limit = 5
        self.print_logging = True

    def add_proposition(self, proposition):
        if self.print_logging:
            print(f"\n[bold cyan]Adding proposition:[/bold cyan] {proposition}")
        if len(self.chunks) == 0:
            if self.print_logging:
                print("[yellow]No existing chunks. Creating a new one.[/yellow]")
            self._create_new_chunk(proposition)
            return

        chunk_id = self._find_relevant_chunk(proposition)
        if chunk_id:
            if self.print_logging:
                print(f"[green]Found relevant chunk {chunk_id}. Adding proposition.[/green]")
            self.chunks[chunk_id]['propositions'].append(proposition)
            self.chunks[chunk_id]['summary'] = self._update_chunk_summary(self.chunks[chunk_id])
            self.chunks[chunk_id]['title'] = self._update_chunk_title(self.chunks[chunk_id])
        else:
            if self.print_logging:
                print("[red]No suitable chunk found. Creating new chunk.[/red]")
            self._create_new_chunk(proposition)

    def _create_new_chunk(self, proposition):
        new_chunk_id = str(uuid.uuid4())[:self.id_truncate_limit]
        summary = self._get_new_chunk_summary(proposition)
        title = self._get_new_chunk_title(summary)
        self.chunks[new_chunk_id] = {
            'chunk_id': new_chunk_id,
            'propositions': [proposition],
            'summary': summary,
            'title': title
        }
        if self.print_logging:
            print(f"[bold magenta]Created new chunk:[/bold magenta] {new_chunk_id} -> {title}")

    def _find_relevant_chunk(self, proposition):
        current_chunk_outline = self.get_chunk_outline()
        PROMPT = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                    Determine whether or not the "Proposition" should belong to any of the existing chunks.

                    A proposition should belong to a chunk of their meaning, direction, or intention are similar.
                    The goal is to group similar propositions and chunks.

                    If you think a proposition should be joined with a chunk, return ONLY the chunk id.
                    If you do not think an item should be joined with an existing chunk, just return "No chunks"

                    Example:
                    Input:
                        - Proposition: "Greg really likes hamburgers"
                        - Current Chunks:
                            - Chunk ID: 2n4l3d
                            - Chunk Name: Places in San Francisco
                            - Chunk Summary: Overview of the things to do with San Francisco Places

                            - Chunk ID: 93833k
                            - Chunk Name: Food Greg likes
                            - Chunk Summary: Lists of the food and dishes that Greg likes
                    Output: 93833k
                    """,
                ),
                ("user", "Current Chunks:\n--Start of current chunks--\n{current_chunk_outline}\n--End of current chunks--"),
                ("user", "Determine if the following statement should belong to one of the chunks outlined:\n{proposition}"),
            ]
        )

        runnable = PROMPT | self.llm | StrOutputParser()
        chunk_found = runnable.invoke({
            "proposition": proposition,
            "current_chunk_outline": current_chunk_outline
        }).strip()

        # If the response is "No chunks" or doesn't match our chunk ID format, return None
        if chunk_found == "No chunks" or len(chunk_found) != self.id_truncate_limit:
            return None

        return chunk_found

    def _get_new_chunk_summary(self, proposition):
        PROMPT = ChatPromptTemplate.from_messages([
            ("system", """
You are the steward of chunks about similar topics.
Summarize the following proposition into one short sentence.
Generalize topics like apples -> food, March -> dates/times.

Example:
Input: Greg likes to eat pizza
Output: This chunk contains information about the types of food Greg likes to eat.

Only return the summary, nothing else.
"""),
            ("user", "Proposition:\n{proposition}")
        ])
        runnable = PROMPT | self.llm
        return runnable.invoke({"proposition": proposition}).content

    def _get_new_chunk_title(self, summary):
        PROMPT = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                    You are the steward of a group of chunks which represent groups of sentences that talk about a similar topic
                    You should generate a very brief few word chunk title which will inform viewers what a chunk group is about.

                    A good chunk title is brief but encompasses what the chunk is about

                    You will be given a summary of a chunk which needs a title

                    Your titles should anticipate generalization. If you get a proposition about apples, generalize it to food.
                    Or month, generalize it to "date and times".

                    Example:
                    Input: Summary: This chunk is about dates and times that the author talks about
                    Output: Date & Times

                    Only respond with the new chunk title, nothing else.
                    """,
                ),
                ("user", "Determine the title of the chunk that this summary belongs to:\n{summary}"),
            ]
        )
        runnable = PROMPT | self.llm
        return runnable.invoke({"summary": summary}).content

    def _update_chunk_summary(self, chunk):
        PROMPT = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                    You are the steward of a group of chunks which represent groups of sentences that talk about a similar topic
                    A new proposition was just added to one of your chunks, you should generate a very brief 1-sentence summary which will inform viewers what a chunk group is about.

                    A good summary will say what the chunk is about, and give any clarifying instructions on what to add to the chunk.

                    You will be given a group of propositions which are in the chunk and the chunks current summary.

                    Your summaries should anticipate generalization. If you get a proposition about apples, generalize it to food.
                    Or month, generalize it to "date and times".

                    Example:
                    Input: Proposition: Greg likes to eat pizza
                    Output: This chunk contains information about the types of food Greg likes to eat.

                    Only respond with the chunk new summary, nothing else.
                    """,
                ),
                ("user", "Chunk's propositions:\n{proposition}\n\nCurrent chunk summary:\n{current_summary}"),
            ]
        )
        runnable = PROMPT | self.llm
        return runnable.invoke({
            "propositions": "\n".join(chunk['propositions']),
            "current_summary": chunk['summary']
        }).content

    def _update_chunk_title(self, chunk):
        PROMPT = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                    You are the steward of a group of chunks which represent groups of sentences that talk about a similar topic
                    A new proposition was just added to one of your chunks, you should generate a very brief updated chunk title which will inform viewers what a chunk group is about.

                    A good title will say what the chunk is about.

                    You will be given a group of propositions which are in the chunk, chunk summary and the chunk title.

                    Your title should anticipate generalization. If you get a proposition about apples, generalize it to food.
                    Or month, generalize it to "date and times".

                    Example:
                    Input: Summary: This chunk is about dates and times that the author talks about
                    Output: Date & Times

                    Only respond with the new chunk title, nothing else.
                    """,
                ),
                ("user", "Chunk's propositions:\n{proposition}\n\nChunk summary:\n{current_summary}\n\nCurrent chunk title:\n{current_title}"),
            ]
        )
        runnable = PROMPT | self.llm
        return runnable.invoke({
            "propositions": "\n".join(chunk['propositions']),
            "current_summary": chunk['summary'],
            "current_title": chunk['title']
        }).content

    def get_chunk_outline(self):
        outline = ""
        for chunk_id, chunk in self.chunks.items():
            outline += f"Chunk ID: {chunk_id}\nChunk Title: {chunk['title']}\nChunk Summary: {chunk['summary']}\n\n"
        return outline

    def get_chunks(self, get_type='dict'):
        """
        This function returns the chunks in the format specified by the 'get_type' parameter.
        If 'get_type' is 'dict', it returns the chunks as a dictionary.
        If 'get_type' is 'list_of_strings', it returns the chunks as a list of strings, where each string is a proposition in the chunk.
        """
        if get_type == 'dict':
            return self.chunks
        if get_type == 'list_of_strings':
            chunks = []
            for chunk_id, chunk in self.chunks.items():
                chunks.append(" ".join([x for x in chunk['propositions']]))
            return chunks

    def pretty_print_chunks(self):
        print (f"\nYou have {len(self.chunks)} chunks\n")
        for chunk_id, chunk in self.chunks.items():
            print(f"Chunk #{chunk['chunk_index']}")
            print(f"Chunk ID: {chunk_id}")
            print(f"Summary: {chunk['summary']}")
            print(f"Propositions:")
            for prop in chunk['propositions']:
                print(f"    -{prop}")
            print("\n\n")

    def pretty_print_chunk_outline(self):
        print ("Chunk Outline\n")
        print(self.get_chunk_outline())


# === PROCESS PDFs ===
def load_and_split_pdfs(pdf_folder):
    print(f"Looking for PDFs in: {pdf_folder}")
    documents = []
    pdf_files = glob.glob(os.path.join(pdf_folder, "*.pdf"))
    print(f"Found {len(pdf_files)} PDF files: {pdf_files}")
    
    if not pdf_files:
        print("Warning: No PDF files found in the specified directory!")
        return documents
        
    loader = PyPDFLoader

    for pdf_path in pdf_files:
        print(f"Loading {pdf_path}...")
        try:
            pdf_loader = loader(pdf_path)
            docs = pdf_loader.load()
            documents.extend(docs)
            print(f"Successfully loaded {len(docs)} pages from {pdf_path}")
        except Exception as e:
            print(f"Error loading {pdf_path}: {str(e)}")

    print(f"Loaded {len(documents)} total documents.")
    return documents

# === SPLIT TEXT ===
def split_documents(documents, chunk_size=500, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    split_docs = splitter.split_documents(documents)
    print(f"Split into {len(split_docs)} chunks.")
    return split_docs

# === UPLOAD TO PINECONE ===
def upload_to_pinecone(chunks, embedding_model, index):
    print("Uploading chunks to Pinecone...")
    vectors = []
    
    for i, chunk in enumerate(tqdm(chunks)):
        # Generate embedding
        embedding = embedding_model.embed_query(chunk.page_content if hasattr(chunk, 'page_content') else chunk)
        
        # Pad the embedding to match Pinecone's dimension requirement
        padded_embedding = embedding + [0.0] * (1536 - len(embedding))
        
        # Create vector with metadata
        vector = {
            'id': f'vec{i}',
            'values': padded_embedding,
            'metadata': {
                'text': chunk.page_content if hasattr(chunk, 'page_content') else chunk,
                'source': 'local'
            }
        }
        vectors.append(vector)
        
        # Upload in batches of 100
        if len(vectors) >= 100:
            index.upsert(vectors=vectors)
            vectors = []
    
    # Upload any remaining vectors
    if vectors:
        index.upsert(vectors=vectors)
    
    print(f"Successfully uploaded {len(chunks)} chunks to Pinecone")

# === MAIN EXECUTION ===
if __name__ == "__main__":
    print("\n#### AGENTIC RAG PIPELINE STARTING ####\n")

    documents = load_and_split_pdfs(PDF_FOLDER)
    split_docs = split_documents(documents)

    print("Chunking documents agentically...")
    chunker = AgenticChunker()
    for doc in split_docs:
        chunker.add_proposition(doc.page_content)

    agentic_chunks = chunker.get_chunks(get_type='list_of_strings')
    print(f"Generated {len(agentic_chunks)} agentic chunks")
    
    # Upload chunks to Pinecone
    upload_to_pinecone(agentic_chunks, embedding_model, pinecone_index)
    
    # final_documents = [Document(page_content=chunk, metadata={"source": "local"}) for chunk in agentic_chunks]

    # print("Creating vectorstore...")
    # vectorstore = create_vectorstore(final_documents, CHROMA_COLLECTION_NAME)
    # retriever = vectorstore.as_retriever()

    # rag_chain = build_rag_chain(retriever)

    # while True:
    #     user_query = input("\nAsk a question about your PDFs (or type 'exit'): ")
    #     if user_query.lower() == "exit":
    #         break
    #     result = rag_chain.invoke(user_query)
    #     print(f"\nAnswer:\n{result}\n")

    
    print("\n#### AGENTIC RAG PIPELINE COMPLETED ####\n")
