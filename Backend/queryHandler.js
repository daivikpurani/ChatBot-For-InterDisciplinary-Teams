import ollama from "ollama";
// import { HuggingFaceEmbeddings } from "@langchain/community/embeddings/huggingface";
// import { PineconeStore } from "@langchain/pinecone";
import { Pinecone } from "@pinecone-database/pinecone";
import dotenv from "dotenv";
import fetch from "node-fetch";
dotenv.config();
// Initialize models
// const model = new ChatOllama({
//   model: process.env.LOCAL_MODEL_NAME,
// });

// let embedding_model;
// try {
//   embedding_model = new HuggingFaceEmbeddings({
//     modelName:
//       process.env.EMBEDDING_MODEL_NAME ||
//       "sentence-transformers/sentence-t5-large",
//     modelKwargs: { device: "cpu" },
//     encodeKwargs: { normalizeEmbeddings: true },
//   });
// } catch (error) {
//   console.error("Error initializing embedding model:", error);
//   console.log("Falling back to default embedding model...");
//   embedding_model = new HuggingFaceEmbeddings({
//     modelName: "sentence-transformers/all-MiniLM-L6-v2",
//     modelKwargs: { device: "cpu" },
//     encodeKwargs: { normalizeEmbeddings: true },
//   });
// }

async function getEmbedding(text) {
  const response = await fetch("http://localhost:11434/api/embeddings", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      model: "llama3.2",
      prompt: text,
    }),
  });

  if (!response.ok) {
    throw new Error(`Ollama API error: ${response.statusText}`);
  }

  const data = await response.json();
  return data.embedding;
}

// Initialize Pinecone
// const vectorStore = new PineconeStore(pineconeIndex, embedding_model);
const pc = new Pinecone({ apiKey: process.env.PINECONE_API_KEY });
const pineconeIndex = pc.index(
  process.env.PINECONE_INDEX_NAME,
  process.env.PINECONE_INDEX_HOST
);

// Query processing functions
const cleanQuery = (query) => {
  // Remove special characters and extra whitespace
  let processedQuery = query.replace(/[^\w\s]/g, " ");
  processedQuery = processedQuery.split(/\s+/).join(" ");

  // Convert to lowercase
  processedQuery = processedQuery.toLowerCase();

  return processedQuery;
};

const improveQuery = async (originalQuery) => {
  const prompt = {
    model: "llama3.2",
    messages: [
      {
        role: "system",
        content: `You are a search query optimizer. Your task is to improve the given search query to get better results.
Follow these rules strictly:
1. DO NOT add concepts or terms that are not directly related to the query
2. DO NOT make assumptions about the context
3. ONLY add synonyms or closely related terms that are directly relevant
4. Keep the core meaning of the original query intact
5. If the query is about a specific term or concept, focus on that term/concept
6. Include both specific and general terms to capture different levels of relevance

Return ONLY the improved query, nothing else.`,
      },
      {
        role: "user",
        content: `Improve this search query: ${originalQuery}`,
      },
    ],
    stream: false,
  };

  const response = await fetch("http://localhost:11434/api/chat", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(prompt),
  });

  if (!response.ok) {
    throw new Error(`Ollama API error: ${response.statusText}`);
  }

  const data = await response.json();
  return data.message.content.trim();
};

const queryIndex = async (query, topK = 10) => {
  // Generate embedding for the query
  const queryEmbedding = await getEmbedding(query);

  // Truncate the embedding to match Pinecone's dimension requirement
  const truncatedEmbedding = queryEmbedding.slice(0, 1536);

  // Query the index
  const results = await pineconeIndex.query({
    vector: truncatedEmbedding,
    topK: topK,
    includeMetadata: true,
    filter: {
      source: "local", // Filter by source if needed
    },
  });

  // Log the results structure for debugging
  console.log("Pinecone results:", JSON.stringify(results, null, 2));

  return results.matches || [];
};

const rerankResults = async (query, results, topK = 5) => {
  const prompt = {
    model: "llama3.2",
    messages: [
      {
        role: "system",
        content: `You are a search result reranker. Your task is to evaluate how well each result matches the query.
Consider:
1. Semantic relevance to the query
2. Completeness of information
3. Specificity to the query topic
4. Clarity and coherence

Return ONLY the indices of the top ${topK} most relevant results in order of relevance.
Example format: "2,0,1" means result 2 is most relevant, then 0, then 1.`,
      },
      {
        role: "user",
        content: `Query: ${query}

Results:
${results
  .map((result, i) => `Result ${i}:\n${result.metadata.text}\n`)
  .join("\n")}

Return the indices of the top ${topK} most relevant results:`,
      },
    ],
    stream: false,
  };

  const response = await fetch("http://localhost:11434/api/chat", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(prompt),
  });

  if (!response.ok) {
    throw new Error(`Ollama API error: ${response.statusText}`);
  }

  const data = await response.json();
  const indices = data.message.content
    .split(",")
    .map((idx) => parseInt(idx.trim()))
    .filter((idx) => !isNaN(idx) && idx >= 0 && idx < results.length);

  return indices.map((idx) => results[idx]);
};

const generateResponse = async (query, results) => {
  // Log the results for debugging
  console.log("Results in generateResponse:", JSON.stringify(results, null, 2));

  const context = results
    .map((result, i) => `Source ${i + 1}:\n${result.metadata.text}`)
    .join("\n\n");

  const prompt = {
    model: "llama3.2",
    messages: [
      {
        role: "system",
        content: `You are a helpful AI assistant. Your task is to generate a clear, concise, and accurate response to the user's query based on the provided sources.
Follow these guidelines:
1. Use only information from the provided sources
2. Write in a clear, conversational tone
3. Structure your response logically
4. If the sources contain conflicting information, acknowledge this
5. If you're unsure about something, say so
6. Keep the response focused and relevant to the query

Format your response as a well-structured paragraph or multiple paragraphs if needed.`,
      },
      {
        role: "user",
        content: `Query: ${query}

Sources:
${context}

Please provide a clear and accurate response to the query based on the sources above:`,
      },
    ],
    stream: false,
  };

  const response = await fetch("http://localhost:11434/api/chat", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(prompt),
  });

  if (!response.ok) {
    throw new Error(`Ollama API error: ${response.statusText}`);
  }

  const data = await response.json();
  return data.message.content.trim();
};

// Main query handler function
export const handleQuery = async (query, topK = 10, rerankK = 5) => {
  try {
    // Clean and improve the query
    const cleanedQuery = cleanQuery(query);
    const improvedQuery = await improveQuery(cleanedQuery);
    console.log("Improved query:", improvedQuery);

    // Query the index directly
    const searchResults = await queryIndex(improvedQuery, topK);

    if (!searchResults || searchResults.length === 0) {
      return {
        reply: "I couldn't find any relevant information to answer your query.",
        improvedQuery,
        retrievedDocs: false,
      };
    }

    // Rerank results
    const rerankedResults = await rerankResults(
      cleanedQuery,
      searchResults,
      rerankK
    );

    if (!rerankedResults || rerankedResults.length === 0) {
      return {
        reply: "I couldn't find any relevant information to answer your query.",
        improvedQuery,
        retrievedDocs: false,
      };
    }

    // Generate response
    const response = await generateResponse(cleanedQuery, rerankedResults);

    return {
      reply: response,
      improvedQuery,
      retrievedDocs: true,
      sources: rerankedResults.map((result) => result.metadata.text),
    };
  } catch (error) {
    console.error("Error processing query:", error);
    throw new Error("Failed to process query");
  }
};
