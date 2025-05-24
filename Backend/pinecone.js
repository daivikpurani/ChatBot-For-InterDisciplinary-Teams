import { Pinecone } from "@pinecone-database/pinecone";
import dotenv from "dotenv";

dotenv.config();

const pinecone = new Pinecone({
  apiKey: process.env.PINECONE_API_KEY
  // environment: process.env.PINECONE_ENV, // e.g., "us-west1-gcp"
});

export default pinecone;
