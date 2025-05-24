import os
import re
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from rich import print

# Load environment variables
load_dotenv()

# === CONFIG ===
LOCAL_MODEL_NAME = "llama3.2"
EMBEDDING_MODEL_NAME = "sentence-transformers/sentence-t5-large"
PINECONE_INDEX_NAME = "agentic-chunks"
TOP_K = 10  # Increased to get more candidates for reranking
RERANK_K = 5  # Number of results to return after reranking
DEFAULT_QUERY = "I am a biology graduate student,explain deepseek in a simple manner"  # Default test query

# === SETUP ===
# Initialize LLM
local_llm = ChatOllama(model=LOCAL_MODEL_NAME)

# Initialize embedding model
try:
    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cpu'},
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
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key=PINECONE_API_KEY)
pinecone_index = pc.Index(PINECONE_INDEX_NAME)

def preprocess_query(query: str) -> str:
    """
    Preprocess the query to improve search results.
    
    Args:
        query (str): Original query
        
    Returns:
        str: Preprocessed query
    """
    # Remove special characters and extra whitespace
    query = re.sub(r'[^\w\s]', ' ', query)
    query = ' '.join(query.split())
    
    # Convert to lowercase
    query = query.lower()
    
    return query

def improve_query(original_query: str) -> str:
    """
    Use LLM to improve the search query by generating a better version.
    
    Args:
        original_query (str): The original user query
        
    Returns:
        str: Improved query
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", """
You are a search query optimizer. Your task is to improve the given search query to get better results.
Follow these rules strictly:
1. DO NOT add concepts or terms that are not directly related to the query
2. DO NOT make assumptions about the context
3. ONLY add synonyms or closely related terms that are directly relevant
4. Keep the core meaning of the original query intact
5. If the query is about a specific term or concept, focus on that term/concept
6. Include both specific and general terms to capture different levels of relevance

Return ONLY the improved query, nothing else.

Examples:
Input: "machine learning"
Output: "machine learning ML artificial intelligence AI algorithms models"

Input: "python programming"
Output: "python programming coding development software engineering"

Input: "what is deepseek"
Output: "deepseek definition explanation overview introduction purpose functionality"
"""),
        ("user", "Improve this search query: {query}")
    ])
    
    chain = prompt | local_llm | StrOutputParser()
    improved_query = chain.invoke({"query": original_query})
    return improved_query.strip()

def rerank_results(query: str, results: list, top_k: int = RERANK_K) -> list:
    """
    Rerank results using LLM to better match the query intent.
    
    Args:
        query (str): Original query
        results (list): List of search results
        top_k (int): Number of results to return after reranking
        
    Returns:
        list: Reranked results
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", """
You are a search result reranker. Your task is to evaluate how well each result matches the query.
Consider:
1. Semantic relevance to the query
2. Completeness of information
3. Specificity to the query topic
4. Clarity and coherence

Return ONLY the indices of the top {top_k} most relevant results in order of relevance.
Example format: "2,0,1" means result 2 is most relevant, then 0, then 1.
"""),
        ("user", """
Query: {query}

Results:
{results}

Return the indices of the top {top_k} most relevant results:
""")
    ])
    
    # Format results for the prompt
    formatted_results = "\n".join([
        f"Result {i}:\n{result.metadata['text']}\n"
        for i, result in enumerate(results)
    ])
    
    chain = prompt | local_llm | StrOutputParser()
    reranked_indices = chain.invoke({
        "query": query,
        "results": formatted_results,
        "top_k": top_k
    })
    
    # Parse the indices and return reranked results
    try:
        indices = [int(idx.strip()) for idx in reranked_indices.split(",")]
        return [results[idx] for idx in indices]
    except:
        # If parsing fails, return original results
        return results[:top_k]

def query_index(query: str, top_k: int = TOP_K):
    """
    Query the Pinecone index and return the most relevant chunks.
    
    Args:
        query (str): The query string to search for
        top_k (int): Number of most relevant chunks to return
        
    Returns:
        list: List of relevant chunks with their metadata and scores
    """
    # Preprocess the query
    processed_query = preprocess_query(query)
    
    # Generate embedding for the query
    query_embedding = embedding_model.embed_query(processed_query)
    
    # Pad the embedding to match Pinecone's dimension requirement
    padded_embedding = query_embedding + [0.0] * (1536 - len(query_embedding))
    
    # Query the index with hybrid search
    results = pinecone_index.query(
        vector=padded_embedding,
        top_k=top_k,
        include_metadata=True,
        filter={
            "source": "local"  # Filter by source if needed
        }
    )
    
    return results

def generate_human_response(query: str, results: list) -> str:
    """
    Generate a human-readable response using the LLM based on the search results.
    
    Args:
        query (str): The original query
        results (list): List of search results
        
    Returns:
        str: Human-readable response
    """
    # Format the results for the prompt
    context = "\n\n".join([
        f"Source {i+1}:\n{result.metadata['text']}"
        for i, result in enumerate(results)
    ])
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """
You are a helpful AI assistant. Your task is to generate a clear, concise, and accurate response to the user's query based on the provided sources.
Follow these guidelines:
1. Use only information from the provided sources
2. Write in a clear, conversational tone
3. Structure your response logically
4. If the sources contain conflicting information, acknowledge this
5. If you're unsure about something, say so
6. Keep the response focused and relevant to the query

Format your response as a well-structured paragraph or multiple paragraphs if needed.
"""),
        ("user", """
Query: {query}

Sources:
{context}

Please provide a clear and accurate response to the query based on the sources above:
""")
    ])
    
    chain = prompt | local_llm | StrOutputParser()
    response = chain.invoke({
        "query": query,
        "context": context
    })
    
    return response.strip()

def format_results(results, original_query: str = None, improved_query: str = None):
    """
    Format the query results in a readable way.
    
    Args:
        results: Query results from Pinecone
        original_query: The original user query
        improved_query: The improved query from LLM
        
    Returns:
        str: Formatted results string
    """
    formatted = "\n=== Query Results ===\n\n"
    
    if original_query and improved_query:
        formatted += f"Original query: {original_query}\n"
        formatted += f"Improved query: {improved_query}\n\n"
    
    # Sort matches by score in descending order
    sorted_matches = sorted(results.matches, key=lambda x: x.score, reverse=True)
    
    # Rerank results using LLM
    reranked_matches = rerank_results(original_query, sorted_matches)
    
    # Generate human-readable response
    human_response = generate_human_response(original_query, reranked_matches)
    
    formatted += "\n=== Generated Response ===\n\n"
    formatted += human_response + "\n\n"
    
    formatted += "\n=== Supporting Sources ===\n\n"
    for i, match in enumerate(reranked_matches, 1):
        formatted += f"Source {i} (Score: {match.score:.4f}):\n"
        formatted += f"Text: {match.metadata['text']}\n"
        formatted += "-" * 80 + "\n"
    
    return formatted

def main():
    print("\n=== Pinecone Query Interface ===\n")
    
    # Run default test query
    print("\nRunning default test query...")
    print(f"Query: {DEFAULT_QUERY}")
    
    print("\nImproving query with LLM...")
    improved_query = improve_query(DEFAULT_QUERY)
    print(f"Improved query: {improved_query}")
    
    print("\nSearching...")
    results = query_index(improved_query)
    
    if results.matches:
        print(format_results(results, original_query=DEFAULT_QUERY, improved_query=improved_query))
    else:
        print("No results found.")
    
    # Interactive query loop
    while True:
        query = input("\nEnter your query (or 'exit' to quit): ")
        if query.lower() == 'exit':
            break
            
        print("\nImproving query with LLM...")
        improved_query = improve_query(query)
        print(f"Improved query: {improved_query}")
        
        print("\nSearching...")
        results = query_index(improved_query)
        
        if results.matches:
            print(format_results(results, original_query=query, improved_query=improved_query))
        else:
            print("No results found.")
    
    print("\nGoodbye!")

if __name__ == "__main__":
    main()
