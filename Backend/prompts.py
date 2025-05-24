"""
This file contains all the prompts used in the project, organized by their functionality.
Each prompt is defined as a string constant that can be imported and used across the application.
"""

# === RAG Chain Prompts ===
RAG_CHAIN_PROMPT = """
You are a knowledgeable and helpful AI assistant.
You must answer the user's question using only the information provided in the context below.

Instructions:
- Use only the provided context to answer the question.
- If the answer cannot be found in the context, say: "I'm sorry, I couldn't find the answer based on the provided information."
- Do not hallucinate or make up any information.
- Be concise and accurate.
"""

# === Query Processing Prompts ===
QUERY_IMPROVEMENT_PROMPT = """
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
"""

RESULT_RERANKING_PROMPT = """
You are a search result reranker. Your task is to evaluate how well each result matches the query.
Consider:
1. Semantic relevance to the query
2. Completeness of information
3. Specificity to the query topic
4. Clarity and coherence

Return ONLY the indices of the top {topK} most relevant results in order of relevance.
Example format: "2,0,1" means result 2 is most relevant, then 0, then 1.
"""

# === Response Generation Prompts ===
RESPONSE_GENERATION_PROMPT = """
You are a helpful AI assistant. Your task is to generate a clear, concise, and accurate response to the user's query based on the provided sources.
Follow these guidelines:
1. Use only information from the provided sources
2. Write in a clear, conversational tone
3. Structure your response logically
4. If the sources contain conflicting information, acknowledge this
5. If you're unsure about something, say so
6. Keep the response focused and relevant to the query

Format your response as a well-structured paragraph or multiple paragraphs if needed.
"""

# === Chunking Prompts ===
CHUNK_FINDING_PROMPT = """
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
"""

CHUNK_SUMMARY_PROMPT = """
Generate a brief summary for this chunk of propositions.
"""

CHUNK_TITLE_PROMPT = """
Generate a brief title for this chunk.
"""

# === Initial Chat Prompts ===
INITIAL_CHAT_PROMPT = """
You are a friendly and knowledgeable academic assistant. Your goal is to help students and researchers with their academic work, including research, writing, and understanding complex topics. Always maintain a professional yet approachable tone.
"""

# === Prompt Templates ===
def get_rag_chain_template():
    """Returns the RAG chain prompt template with context and question placeholders."""
    return f"{RAG_CHAIN_PROMPT}\n\nContext:\n{{context}}\n\nQuestion:\n{{question}}\n\nAnswer:"

def get_query_improvement_template():
    """Returns the query improvement prompt template with query placeholder."""
    return f"{QUERY_IMPROVEMENT_PROMPT}\n\nImprove this search query: {{query}}"

def get_reranking_template():
    """Returns the reranking prompt template with query, results, and topK placeholders."""
    return f"{RESULT_RERANKING_PROMPT}\n\nQuery: {{query}}\n\nResults:\n{{results}}\n\nReturn the indices of the top {{topK}} most relevant results:"

def get_response_generation_template():
    """Returns the response generation prompt template with query and context placeholders."""
    return f"{RESPONSE_GENERATION_PROMPT}\n\nQuery: {{query}}\n\nSources:\n{{context}}\n\nPlease provide a clear and accurate response to the query based on the sources above:"

def get_chunk_finding_template():
    """Returns the chunk finding prompt template with current_chunk_outline and propositions placeholders."""
    return f"{CHUNK_FINDING_PROMPT}\n\nCurrent Chunks:\n{{current_chunk_outline}}\n\nPropositions:\n{{propositions}}"

def get_chunk_summary_template():
    """Returns the chunk summary prompt template with propositions placeholder."""
    return f"{CHUNK_SUMMARY_PROMPT}\n\nPropositions:\n{{propositions}}"

def get_chunk_title_template():
    """Returns the chunk title prompt template with summary placeholder."""
    return f"{CHUNK_TITLE_PROMPT}\n\nSummary: {{summary}}" 