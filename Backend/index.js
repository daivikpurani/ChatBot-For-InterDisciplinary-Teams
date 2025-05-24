import express from "express";
import cors from "cors";
import dotenv from "dotenv";
import { handleQuery } from "./queryHandler.js";

dotenv.config();

const app = express();
app.use(cors());
app.use(express.json());

// Set up initial questions
const initial_questions = [
  {
    role: "system",
    content:
      "You are a friendly and knowledgeable academic assistant. Your goal is to help students and researchers with their academic work, including research, writing, and understanding complex topics. Always maintain a professional yet approachable tone.",
  },
  {
    role: "assistant",
    content:
      "Hello! I'm your academic research assistant. Before we begin, could you please tell me your name and what academic field or subject you're working on? This will help me provide more relevant and targeted assistance for your research needs.",
  },
];

app.post("/api/chat", async (req, res) => {
  try {
    const { message, is_new_chat } = req.body;
    console.log("Received message:", message);

    // Handle the query using query handler
    const result = await handleQuery(message);
    console.log("Query result:", result);

    // Send the response
    res.status(200).json({
      success: true,
      data: {
        reply: result.reply,
        improvedQuery: result.improvedQuery,
        retrievedDocs: result.retrievedDocs,
        sources: result.sources,
      },
    });
  } catch (error) {
    console.error("API error:", error);
    res.status(500).json({
      success: false,
      error: "Failed to process request",
      message: error.message,
    });
  }
});

const PORT = process.env.PORT || 8000;
app.listen(PORT, () => console.log(`Server running on port ${PORT}`));
