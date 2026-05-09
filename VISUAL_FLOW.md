# RAG Lab: Visual Step-by-Step Flow

## Complete RAG System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        RAG PIPELINE OVERVIEW                            │
└─────────────────────────────────────────────────────────────────────────┘

                              USER QUESTION
                                   ↓
                         ┌─────────────────────┐
                         │  Query "penalties   │
                         │  for AI violations" │
                         └────────┬────────────┘
                                  ↓
                    ┌──────────────────────────────┐
         STEP 1     │ Convert Query to Embedding  │
                    │ (OpenAI API)               │
                    │ "penalties..." → [0.1,0.2..] │
                    └────────────┬─────────────────┘
                                 ↓
              ┌──────────────────────────────────────────┐
    STEP 2    │ Search Pinecone Vector Database        │
              │ Find 5 most similar documents          │
              │                                        │
              │ Similarity scores:                     │
              │   1. Document A: 0.89                  │
              │   2. Document B: 0.86                  │
              │   3. Document C: 0.84                  │
              │   4. Document D: 0.82                  │
              │   5. Document E: 0.80                  │
              └────────┬─────────────────────────────────┘
                       ↓
            ┌──────────────────────────────────────┐
            │     DECISION POINT                   │
            │  Use Reranking?                      │
            └────┬────────────────────────┬────────┘
                 │                        │
                 NO                       YES
                 │                        │
                 ↓                        ↓
        ┌───────────────┐    ┌─────────────────────────┐
        │Return top 3   │    │ STEP 3: Reranking      │
        │as-is          │    │ (Cohere API)           │
        │               │    │                        │
        │ 1. Doc A      │    │ Cohere evaluates:      │
        │ 2. Doc B      │    │ "How RELEVANT is each  │
        │ 3. Doc C      │    │  to the query?"        │
        │               │    │                        │
        │ ❌ Order by   │    │ New scores:            │
        │   similarity  │    │   1. Doc C: 0.92 ⭐    │
        │               │    │   2. Doc A: 0.87       │
        │               │    │   3. Doc E: 0.85       │
        └───────────────┘    │                        │
                             │ ✅ Order by relevance │
                             └──────────┬────────────┘
                                        ↓
                            ┌──────────────────────┐
                            │  Return Top 3        │
                            │  Reranked Results    │
                            │                      │
                            │  1. Doc C (0.92)    │
                            │  2. Doc A (0.87)    │
                            │  3. Doc E (0.85)    │
                            └──────────┬───────────┘
                                       ↓
                            ┌──────────────────────┐
                            │ Display to User      │
                            │ (Better results!)    │
                            └──────────────────────┘
```

---

## Step-by-Step Process (What Happens Behind the Scenes)

### STEP 1: Data Loading & Preparation

```
PDF File
  ├─ Extract text using pdfplumber
  ├─ Identify structure (headings, sections)
  └─ Create metadata
       ├─ Text: "Chapter 1: AI Penalties"
       ├─ Source: "legal_document"
       ├─ Level: "article"
       └─ Page: 5

    ↓ RESULT: List of documents with metadata ↓
    
    [
      {
        "text": "High-risk AI...",
        "source_type": "legal_document",
        "hierarchy_level": "article",
        "page": 10
      },
      {
        "text": "Prohibited practices...",
        "source_type": "legal_document",
        "hierarchy_level": "article",
        "page": 15
      },
      ...
    ]
```

### STEP 2: Embedding Generation

```
Text Documents              OpenAI API              Embedding Vectors
  │                            │                         │
  ├─ "High-risk AI..."  ────→ embedding() ────→ [0.123, -0.456, 0.789...]
  │                            │                         │
  ├─ "Prohibited..."    ────→ embedding() ────→ [-0.234, 0.567, 0.123...]
  │                            │                         │
  └─ "Penalties..."     ────→ embedding() ────→ [0.789, 0.234, -0.456...]

STORED IN: Pinecone Vector Database
```

### STEP 3: Vector Storage in Pinecone

```
Pinecone Vector Index
┌─────────────────────────────────────────────────┐
│ ID      │ Vector              │ Metadata        │
├─────────┼─────────────────────┼─────────────────┤
│ doc-001 │ [0.123, -0.456...] │ {"text": "...", │
│         │                     │  "page": 10}    │
├─────────┼─────────────────────┼─────────────────┤
│ doc-002 │ [-0.234, 0.567...] │ {"text": "...", │
│         │                     │  "page": 15}    │
├─────────┼─────────────────────┼─────────────────┤
│ doc-003 │ [0.789, 0.234...]  │ {"text": "...", │
│         │                     │  "page": 20}    │
└─────────┴─────────────────────┴─────────────────┘
```

### STEP 4: Query & Search

```
User Question: "What are the penalties for AI violations?"
                          ↓
                    OpenAI Embedding
                          ↓
    Query Vector: [0.110, -0.440, 0.750...]
                          ↓
              Pinecone Similarity Search
                          ↓
        Find most similar vectors using math:
        
        Similarity = how close vectors are
        (Higher = more similar)
        
        Results:
        ┌──────────────────────────────────┐
        │ 1. doc-047: 0.89 (very similar)  │
        │ 2. doc-103: 0.86 (similar)       │
        │ 3. doc-215: 0.84 (similar)       │
        │ 4. doc-089: 0.82 (somewhat)      │
        │ 5. doc-142: 0.80 (somewhat)      │
        └──────────────────────────────────┘
```

### STEP 5: Reranking (Optional but Powerful)

```
Raw Search Results (from Pinecone)
                ↓
        ┌─────────────────────┐
        │  Cohere Rerank API  │
        │                     │
        │ "Given the query:   │
        │  'What are the      │
        │   penalties...'     │
        │                     │
        │  How relevant is    │
        │  each document?"    │
        └────────┬────────────┘
                 ↓
      ┌──────────────────────────┐
      │ Cohere analyzes:        │
      │                         │
      │ Doc-047:                │
      │   ✓ Mentions penalties  │
      │   ✓ About AI violations │
      │   ✓ Legally specific    │
      │   → Score: 0.92 ⭐      │
      │                         │
      │ Doc-103:                │
      │   ✓ About AI topics     │
      │   ✗ Generic info        │
      │   ✗ No penalties        │
      │   → Score: 0.65        │
      │                         │
      │ Doc-215:                │
      │   ✓ Mentions penalties  │
      │   ✗ For fraud, not AI   │
      │   → Score: 0.58        │
      └──────────────────────────┘
                ↓
      ┌──────────────────────────┐
      │ RERANKED RESULTS:       │
      │ (Ordered by relevance)   │
      │                         │
      │ 1. Doc-047: 0.92 ⭐      │
      │ 2. Doc-215: 0.58        │
      │ 3. Doc-103: 0.65        │
      │                         │
      │ ✅ Better quality!      │
      └──────────────────────────┘
```

---

## Key Differences: Similarity vs. Relevance

```
SIMILARITY (Embedding-based)
├─ Mathematical: How close are the vectors?
├─ Fast: Uses pre-computed embeddings
├─ Cheap: No extra API calls
├─ Can be wrong: Similar ≠ Relevant
│
│ Example:
│ Query: "penalties for AI"
│ Doc A: "AI ethics review process" (0.85 similarity)
│         ↑ Looks similar but not about penalties!
│ Doc B: "Fines for non-compliance" (0.78 similarity)
│         ↑ Less similar but actually relevant!

RELEVANCE (Reranking-based)
├─ Semantic: Does this answer the question?
├─ Slower: Requires API call per query
├─ Costs money: Cohere charges per rerank
├─ More accurate: Actually answers the question
│
│ Example (after reranking):
│ Doc B: 0.92 relevance ✅ (moves to top!)
│ Doc A: 0.45 relevance ✗ (moves down)
```

---

## Metadata Filtering Examples

```
Without Metadata Filtering:
┌─────────────────────────────────────┐
│ Search results for "penalties"      │
├─────────────────────────────────────┤
│ ✓ Legal doc: "AI Act penalties"     │
│ ✓ Legal doc: "GDPR penalties"       │
│ ✓ Podcast: "Interview about fines"  │
│ ✓ Blog: "Why penalties matter"      │
│ ✓ Podcast: "Q&A on consequences"    │
└─────────────────────────────────────┘

With Metadata Filtering (legal_document only):
┌─────────────────────────────────────┐
│ Search results for "penalties"      │
├─────────────────────────────────────┤
│ ✓ Legal doc: "AI Act penalties"     │
│ ✓ Legal doc: "GDPR penalties"       │
│ ✗ Podcast: (filtered out)           │
│ ✗ Blog: (filtered out)              │
│ ✗ Podcast: (filtered out)           │
└─────────────────────────────────────┘
     ↑ More focused results!
```

---

## Timeline: What Happens When You Search

```
⏱️  0.0s  User types: "What are AI penalties?"
          └─ Query ready

⏱️  0.1s  System converts to embedding
          └─ Query embedded (OpenAI)

⏱️  0.2s  Search in Pinecone
          └─ Find 5 similar documents
          └─ Results with similarity scores

⏱️  0.3s  [NO RERANKING] Return results ✅
          
          [WITH RERANKING] Send to Cohere ↓

⏱️  1.5s  Cohere evaluates relevance
          └─ Analyzes query vs each document
          
⏱️  1.6s  Return reranked results ✅

⏱️  1.7s  User sees final results
          └─ Much better ordering!

TOTAL TIME:
- Without reranking: ~0.3 seconds ⚡
- With reranking: ~1.5 seconds (3-5x slower but much better)
```

---

## Why Reranking Helps

### Scenario: Searching "penalties for AI violations"

```
PINECONE (Similarity-based):
┌──────────────────────────────────────────────────────┐
│ Rank │ Score │ Content                              │
├──────┼───────┼──────────────────────────────────────┤
│  1   │ 0.87  │ "GDPR penalties and fines..."        │
│      │       │ (High similarity to keyword,        │
│      │       │  but about GDPR, not AI Act)        │
│      │       │ ❌ WRONG                            │
├──────┼───────┼──────────────────────────────────────┤
│  2   │ 0.85  │ "Transparency requirements..."      │
│      │       │ (Mentions penalties, but not main   │
│      │       │  topic)                            │
│      │       │ ⚠️ SOMEWHAT RELEVANT              │
├──────┼───────┼──────────────────────────────────────┤
│  3   │ 0.82  │ "AI penalties under EU AI Act"      │
│      │       │ (This is what we want!)             │
│      │       │ ✅ RIGHT, BUT RANKED 3RD!         │
└──────────────────────────────────────────────────────┘

COHERE (Reranking):
┌──────────────────────────────────────────────────────┐
│ Rank │ Score │ Content                              │
├──────┼───────┼──────────────────────────────────────┤
│  1   │ 0.92  │ "AI penalties under EU AI Act"      │
│      │       │ (Perfect match!)                    │
│      │       │ ✅ BEST FIRST                      │
├──────┼───────┼──────────────────────────────────────┤
│  2   │ 0.78  │ "Transparency requirements..."      │
│      │       │ (Related but not primary)           │
│      │       │ ✅ CORRECT POSITION                │
├──────┼───────┼──────────────────────────────────────┤
│  3   │ 0.45  │ "GDPR penalties and fines..."       │
│      │       │ (Different law)                     │
│      │       │ ✅ CORRECT POSITION                │
└──────────────────────────────────────────────────────┘
```

---

## Implementation Checklist

```
□ STEP 1: Load & Prepare Documents
  └─ Extract from PDF
  └─ Add metadata
  └─ Verify data loaded

□ STEP 2: Generate Embeddings
  └─ Call OpenAI API
  └─ Attach to documents
  └─ Verify embeddings created

□ STEP 3: Upload to Pinecone
  └─ Format data correctly
  └─ Upload vectors
  └─ Verify in Pinecone

□ STEP 4: Test Basic Search
  └─ Create search function
  └─ Test with sample query
  └─ See similarity scores

□ STEP 5: Implement Reranking
  └─ Initialize Cohere client
  └─ Create rerank function
  └─ Test on search results

□ STEP 6: Build Complete Pipeline
  └─ Combine search + reranking
  └─ Make toggle-able
  └─ Test both paths

□ STEP 7: Evaluate Performance
  └─ Test multiple queries
  └─ Compare results
  └─ Create comparison table

□ STEP 8: Document Findings
  └─ Write lab_summary.md
  └─ Create README.md
  └─ Save results.csv
```

---

## Comparison at a Glance

```
┌─────────────────────┬──────────────────────┬──────────────────────┐
│ Feature             │ WITHOUT Reranking    │ WITH Reranking       │
├─────────────────────┼──────────────────────┼──────────────────────┤
│ Speed               │ ⚡ 0.3 seconds      │ ⏱️  1.5 seconds      │
│ Cost per query      │ ~$0 (free)           │ ~$0.01 (cheap)       │
│ Quality             │ 😐 Good              │ 😊 Excellent         │
│ When to use         │ Fast searches        │ Important results    │
│ Ranking method      │ Mathematical         │ Semantic/AI          │
│ Can be wrong        │ Sometimes ❌         │ Rarely ✅            │
│ Good for           │ Simple queries       │ Complex queries      │
└─────────────────────┴──────────────────────┴──────────────────────┘
```

---

