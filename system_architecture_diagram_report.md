# System Architecture Diagram

```mermaid
graph TB
    subgraph Frontend
        UI[Chat Interface]
        State[State Management]
        UI --> State
    end

    subgraph Backend
        subgraph DataProcessing
            PDF[PDF Processing]
            Text[Text Extraction]
            Clean[Text Cleaning]
            PDF --> Text --> Clean
        end

        subgraph VectorStore
            Embed[Embedding Generation]
            Store[Vector Database]
            Clean --> Embed --> Store
        end

        subgraph QueryProcessing
            Query[Query Handler]
            Search[Semantic Search]
            Context[Context Management]
            Query --> Search --> Context
        end

        subgraph PromptSystem
            Prompt[Prompt Management]
            Template[Template Engine]
            Prompt --> Template
        end
    end

    subgraph Integration
        API[API Gateway]
        WS[WebSocket Server]
        Auth[Authentication]
        Cache[Cache Layer]
    end

    %% Frontend to Backend connections
    UI --> API
    UI --> WS
    State --> API

    %% Backend internal connections
    Search --> Store
    Context --> Store
    Template --> Query

    %% Integration connections
    API --> Auth
    API --> Cache
    WS --> Auth

    %% External Services
    subgraph ExternalServices
        Pinecone[(Pinecone DB)]
        LLM[Language Model]
    end

    Store --> Pinecone
    Embed --> LLM
    Search --> LLM

    %% Styling with report-friendly colors
    classDef frontend fill:#2E86C1,stroke:#1A5276,color:white
    classDef backend fill:#E74C3C,stroke:#922B21,color:white
    classDef integration fill:#27AE60,stroke:#196F3D,color:white
    classDef external fill:#F39C12,stroke:#9A7D0A,color:white

    class UI,State frontend
    class DataProcessing,VectorStore,QueryProcessing,PromptSystem backend
    class API,WS,Auth,Cache integration
    class Pinecone,LLM external
```

## Component Descriptions

### Frontend Components

- **Chat Interface**: User-facing chat UI with real-time interaction
- **State Management**: Handles application state and user session data

### Backend Components

- **Data Processing**

  - PDF Processing: Handles document ingestion
  - Text Extraction: Converts PDFs to text
  - Text Cleaning: Prepares text for embedding

- **Vector Store**

  - Embedding Generation: Converts text to vectors
  - Vector Database: Stores and manages embeddings

- **Query Processing**

  - Query Handler: Processes user queries
  - Semantic Search: Performs vector similarity search
  - Context Management: Maintains conversation context

- **Prompt System**
  - Prompt Management: Handles system prompts
  - Template Engine: Manages response templates

### Integration Layer

- **API Gateway**: Manages all API requests
- **WebSocket Server**: Handles real-time communication
- **Authentication**: Manages security and access
- **Cache Layer**: Optimizes response times

### External Services

- **Pinecone DB**: Vector database service
- **Language Model**: Provides embedding and semantic understanding
