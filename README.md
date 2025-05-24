# Academic Research Assistant

An intelligent academic research assistant that helps students and researchers with their academic work through advanced AI capabilities. This project implements a sophisticated RAG (Retrieval-Augmented Generation) system with smart content organization and query processing.

## Features

- **RAG (Retrieval-Augmented Generation) System**:
  - Context-aware responses using a sophisticated retrieval system
  - Integration with vector store for efficient information retrieval
  - Smart content chunking and organization
- **Query Processing**:
  - Query improvement and optimization
  - Result reranking for better relevance
  - Natural language understanding
- **Smart Chunking**:
  - Intelligent content organization
  - Automatic chunk finding and summarization
  - Title generation for content chunks
- **Response Generation**:
  - Clear, concise, and accurate responses based on provided sources
  - Context-aware answer generation
  - Source attribution and verification

## Project Structure

```
.
├── Backend/
│   ├── prompts.py          # AI prompts and templates
│   ├── index.js            # Main server entry point
│   ├── queryHandler.js     # Query processing logic
│   ├── pinecone.js         # Vector store integration
│   ├── vectorstore/        # Vector database storage
│   ├── scrapeddata/        # Processed data storage
│   ├── pdfs/              # PDF document storage
│   └── backupPDF/         # PDF backup storage
├── Frontend/
│   └── chatbot-ui/        # User interface components
└── Data Scraping/         # Data collection and processing
```

## Prerequisites

- Node.js (v14 or higher)
- Python 3.8+
- Pinecone account and API key
- OpenAI API key

## Setup and Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/academic-research-assistant.git
cd academic-research-assistant
```

2. Install Backend dependencies:

```bash
cd Backend
npm install
```

3. Install Frontend dependencies:

```bash
cd Frontend/chatbot-ui
npm install
```

4. Set up environment variables:

```bash
# In Backend directory
cp .env.example .env
```

Required environment variables:

```
OPENAI_API_KEY=your_openai_api_key
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_ENVIRONMENT=your_pinecone_environment
```

## Usage

1. Start the Backend server:

```bash
cd Backend
npm start
```

2. Start the Frontend development server:

```bash
cd Frontend/chatbot-ui
npm run dev
```

3. Access the application at `http://localhost:3000`

### Using the Research Assistant

1. Enter your research question in the chat interface
2. The system will:
   - Process and optimize your query
   - Search through the knowledge base
   - Generate a comprehensive response
   - Provide relevant sources and citations

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- OpenAI for providing the GPT models
- Pinecone for vector database services
- All contributors who have helped shape this project
