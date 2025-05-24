const dotenv = require("dotenv");
const { Pinecone } = require("@pinecone-database/pinecone");
const { HuggingFaceInference } = require("@langchain/community/embeddings/hf");
const { ChatOllama } = require("@langchain/community/chat_models/ollama");
const { ChatPromptTemplate } = require("@langchain/core/prompts");
const { StringOutputParser } = require("@langchain/core/output_parsers");

// Load environment variables
dotenv.config();

// === CONFIG ===
const LOCAL_MODEL_NAME = "llama3.2";
const EMBEDDING_MODEL_NAME = "sentence-transformers/sentence-t5-large";
const PINECONE_INDEX_NAME = "agentic-chunks";
const TOP_K = 10; // Increased to get more candidates for reranking
const RERANK_K = 5; // Number of results to return after reranking
const DEFAULT_QUERY =
  "I am a biology graduate student,explain deepseek in a simple manner"; // Default test query

// === SETUP ===
// Initialize LLM
const localLLM = new ChatOllama({ model: LOCAL_MODEL_NAME });

// Initialize embedding model
let embeddingModel;
try {
  embeddingModel = new HuggingFaceInference({
    modelName: EMBEDDING_MODEL_NAME,
    modelKwargs: { device: "cpu" },
    encodeKwargs: { normalizeEmbeddings: true },
  });
} catch (e) {
  console.error("Error initializing embedding model:", e);
  console.log("Falling back to default embedding model...");
  embeddingModel = new HuggingFaceInference({
    modelName: "sentence-transformers/all-MiniLM-L6-v2",
    modelKwargs: { device: "cpu" },
    encodeKwargs: { normalizeEmbeddings: true },
  });
}

// Initialize Pinecone
const PINECONE_API_KEY = process.env.PINECONE_API_KEY;
const pc = new Pinecone({ apiKey: PINECONE_API_KEY });
const pineconeIndex = pc.Index(PINECONE_INDEX_NAME);

function preprocessQuery(query) {
  // Remove special characters and extra whitespace
  query = query.replace(/[^\w\s]/g, " ");
  query = query.replace(/\s+/g, " ").trim();

  // Convert to lowercase
  query = query.toLowerCase();

  return query;
}

async function improveQuery(originalQuery) {
  const prompt = ChatPromptTemplate.fromMessages([
    [
      "system",
      `
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
`,
    ],
    ["user", "Improve this search query: {query}"],
  ]);

  const chain = prompt.pipe(localLLM).pipe(new StringOutputParser());
  const improvedQuery = await chain.invoke({ query: originalQuery });
  return improvedQuery.trim();
}

async function rerankResults(query, results, topK = RERANK_K) {
  const prompt = ChatPromptTemplate.fromMessages([
    [
      "system",
      `
You are a search result reranker. Your task is to evaluate how well each result matches the query.
Consider:
1. Semantic relevance to the query
2. Completeness of information
3. Specificity to the query topic
4. Clarity and coherence

Return ONLY the indices of the top {topK} most relevant results in order of relevance.
Example format: "2,0,1" means result 2 is most relevant, then 0, then 1.
`,
    ],
    [
      "user",
      `
Query: {query}

Results:
{results}

Return the indices of the top {topK} most relevant results:
`,
    ],
  ]);

  // Format results for the prompt
  const formattedResults = results
    .map((result, i) => `Result ${i}:\n${result.metadata.text}\n`)
    .join("\n");

  const chain = prompt.pipe(localLLM).pipe(new StringOutputParser());
  const rerankedIndices = await chain.invoke({
    query,
    results: formattedResults,
    topK,
  });

  // Parse the indices and return reranked results
  try {
    const indices = rerankedIndices
      .split(",")
      .map((idx) => parseInt(idx.trim()));
    return indices.map((idx) => results[idx]);
  } catch {
    // If parsing fails, return original results
    return results.slice(0, topK);
  }
}

async function queryIndex(query, topK = TOP_K) {
  // Preprocess the query
  const processedQuery = preprocessQuery(query);

  // Generate embedding for the query
  const queryEmbedding = await embeddingModel.embedQuery(processedQuery);

  // Pad the embedding to match Pinecone's dimension requirement
  const paddedEmbedding = [
    ...queryEmbedding,
    ...Array(1536 - queryEmbedding.length).fill(0.0),
  ];

  // Query the index with hybrid search
  const results = await pineconeIndex.query({
    vector: paddedEmbedding,
    topK,
    includeMetadata: true,
    filter: {
      source: "local", // Filter by source if needed
    },
  });

  return results;
}

async function generateHumanResponse(query, results) {
  // Format the results for the prompt
  const context = results
    .map((result, i) => `Source ${i + 1}:\n${result.metadata.text}`)
    .join("\n\n");

  const prompt = ChatPromptTemplate.fromMessages([
    [
      "system",
      `
You are a helpful AI assistant. Your task is to generate a clear, concise, and accurate response to the user's query based on the provided sources.
Follow these guidelines:
1. Use only information from the provided sources
2. Write in a clear, conversational tone
3. Structure your response logically
4. If the sources contain conflicting information, acknowledge this
5. If you're unsure about something, say so
6. Keep the response focused and relevant to the query

Format your response as a well-structured paragraph or multiple paragraphs if needed.
`,
    ],
    [
      "user",
      `
Query: {query}

Sources:
{context}

Please provide a clear and accurate response to the query based on the sources above:
`,
    ],
  ]);

  const chain = prompt.pipe(localLLM).pipe(new StringOutputParser());
  const response = await chain.invoke({
    query,
    context,
  });

  return response.trim();
}

async function formatResults(
  results,
  originalQuery = null,
  improvedQuery = null
) {
  let formatted = "\n=== Query Results ===\n\n";

  if (originalQuery && improvedQuery) {
    formatted += `Original query: ${originalQuery}\n`;
    formatted += `Improved query: ${improvedQuery}\n\n`;
  }

  // Sort matches by score in descending order
  const sortedMatches = results.matches.sort((a, b) => b.score - a.score);

  // Rerank results using LLM
  const rerankedMatches = await rerankResults(originalQuery, sortedMatches);

  // Generate human-readable response
  const humanResponse = await generateHumanResponse(
    originalQuery,
    rerankedMatches
  );

  formatted += "\n=== Generated Response ===\n\n";
  formatted += humanResponse + "\n\n";

  formatted += "\n=== Supporting Sources ===\n\n";
  rerankedMatches.forEach((match, i) => {
    formatted += `Source ${i + 1} (Score: ${match.score.toFixed(4)}):\n`;
    formatted += `Text: ${match.metadata.text}\n`;
    formatted += "-".repeat(80) + "\n";
  });

  return formatted;
}

async function main() {
  console.log("\n=== Pinecone Query Interface ===\n");

  // Run default test query
  console.log("\nRunning default test query...");
  console.log(`Query: ${DEFAULT_QUERY}`);

  console.log("\nImproving query with LLM...");
  const improvedQuery = await improveQuery(DEFAULT_QUERY);
  console.log(`Improved query: ${improvedQuery}`);

  console.log("\nSearching...");
  const results = await queryIndex(improvedQuery);

  if (results.matches.length > 0) {
    console.log(await formatResults(results, DEFAULT_QUERY, improvedQuery));
  } else {
    console.log("No results found.");
  }

  // Interactive query loop
  const readline = require("readline").createInterface({
    input: process.stdin,
    output: process.stdout,
  });

  const askQuery = () => {
    readline.question(
      "\nEnter your query (or 'exit' to quit): ",
      async (query) => {
        if (query.toLowerCase() === "exit") {
          readline.close();
          console.log("\nGoodbye!");
          return;
        }

        console.log("\nImproving query with LLM...");
        const improvedQuery = await improveQuery(query);
        console.log(`Improved query: ${improvedQuery}`);

        console.log("\nSearching...");
        const results = await queryIndex(improvedQuery);

        if (results.matches.length > 0) {
          console.log(await formatResults(results, query, improvedQuery));
        } else {
          console.log("No results found.");
        }

        askQuery();
      }
    );
  };

  askQuery();
}

if (require.main === module) {
  main().catch(console.error);
}

module.exports = {
  queryIndex,
  improveQuery,
  formatResults,
  generateHumanResponse,
};
