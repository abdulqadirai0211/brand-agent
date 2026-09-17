                    ASK REQUEST
                        │
                        ▼
              ┌──────────────────┐
              │   LangGraph Agent │
              │   Orchestration   │
              └────────┬─────────┘
                       │
             ┌─────────▼─────────┐
             │ Retrieval Pipeline │
             └─────────┬─────────┘
                       │
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
     Dense           BM25           Metadata
    Pinecone        Lexical          Filter
        │              │
        └──────┬───────┘
               ▼
             RRF
               │
               ▼
             Reranker
               │
               ▼
           Top ~5 chunks
               │
               ▼
          LLM Relevance
             Grading
               │
       ┌───────┴────────┐
       │                │
    Relevant         Not relevant
       │                │
       ▼                ▼
    Answer          Rewrite Query
       │                │
       │                ▼
       │             Retrieve
       │                │
       │          ┌─────┴─────┐
       │       relevant      none
       │          │            │
       └──────────┘            ▼
                         Not in corpus