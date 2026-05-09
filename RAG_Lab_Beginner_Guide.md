# RAG Lab: Complete Beginner's Guide
## Relevance Scoring & Rerankers for Legal Documents

---

## 🎯 **What This Lab Is About (In Simple Terms)**

Imagine you're building a **search engine for legal documents** (like EU AI Act) and podcast transcripts.

**The Problem:**
- Simple search finds documents that look similar to your question but might not be the MOST relevant
- It's like Google returning 10 results when only 2 are actually what you need

**The Solution (What we're building):**
1. **Basic Search** → Find similar documents using embeddings
2. **Reranking** → Re-evaluate those results to put the BEST ones first
3. **Metadata Filtering** → Filter by type (is it from podcast? legal document?)

**Real-world example:**
- User asks: "What are EU AI Act penalties for high-risk systems?"
- Basic search might return 100 results
- Reranker filters to 5 best results
- You get a better answer! ✅

---

## 📋 **What You've Already Done**

✅ Created `.env` file with API keys  
✅ Have OpenAI API key  
✅ Have Cohere API key (for reranking)  
✅ Have Pinecone API key (vector database)  

**Great! You're ready to start.**

---

## 🚀 **Step-by-Step Breakdown**

### **STEP 1: Setup & Loading Data**
**What to do:** Load your documents and prepare them for the system

#### 1.1 - Import all necessary libraries
```python
# At the very top of your notebook, add:
import os
from dotenv import load_dotenv
import openai
import json
from typing import List, Dict, Any

# Load environment variables
load_dotenv()

# Set up API keys
openai.api_key = os.getenv("OPENAI_API_KEY")
cohere_api_key = os.getenv("COHERE_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
```

**What this does:**
- Loads your `.env` file containing all API keys
- Makes them available to use in your code
- Sets up OpenAI (for embeddings and LLM)
- Sets up Cohere (for reranking)
- Sets up Pinecone (for storing vectors)

#### 1.2 - Load your documents
You already have code that extracts from PDF. Now organize it:

```python
# Your PDF extraction code (already in notebook)
chunks = extract_structure('eu_ai_act.pdf')  # This extracts PDF structure

# Convert to simple format
documents = []
for chunk in chunks:
    doc = {
        "text": chunk["heading"],          # The main content
        "source_type": "legal",             # EU AI Act = legal document
        "section": chunk["level"],          # rule, section, article
        "page": chunk["page"]               # which page it's from
    }
    documents.append(doc)

print(f"Loaded {len(documents)} documents")
print("Sample document:", documents[0])  # See what it looks like
```

**What this does:**
- Takes extracted content from PDF
- Adds metadata (what type of document, which section, page number)
- Creates a simple list we can work with

---

### **STEP 2: Create Embeddings (Convert Text to Numbers)**
**What to do:** Turn your text into numbers that computers can understand

#### 2.1 - Generate embeddings using OpenAI

```python
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_embeddings(texts: List[str]) -> List[List[float]]:
    """
    Convert text to embeddings (numbers).
    
    Think of it as: 
    "EU AI Act penalties" → [0.123, -0.456, 0.789, ...]
    """
    response = client.embeddings.create(
        input=texts,
        model="text-embedding-3-small"  # Fast and cheap model
    )
    return [item.embedding for item in response.data]

# Generate embeddings for all documents
texts = [doc["text"] for doc in documents]
print(f"Generating embeddings for {len(texts)} documents...")
embeddings = get_embeddings(texts)

# Add embeddings to documents
for doc, embedding in zip(documents, embeddings):
    doc["embedding"] = embedding

print(f"✅ Created {len(embeddings)} embeddings")
print(f"Each embedding has {len(embeddings[0])} dimensions")
```

**What this does:**
- Takes each document's text
- Sends it to OpenAI
- Gets back a list of numbers (embedding) that represents that text
- Each document now has a number representation that can be compared

---

### **STEP 3: Store in Vector Database (Pinecone)**
**What to do:** Upload your embeddings to Pinecone so you can search them

#### 3.1 - Connect to Pinecone and upload

```python
from pinecone import Pinecone

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Get or create index
index_name = "legal-rag-index"
index = pc.Index(index_name)

# Prepare data for Pinecone (format it correctly)
vectors_to_upsert = []
for i, doc in enumerate(documents):
    vector = {
        "id": f"doc-{i}",
        "values": doc["embedding"],
        "metadata": {
            "text": doc["text"],
            "source_type": doc["source_type"],
            "section": doc["section"],
            "page": doc["page"]
        }
    }
    vectors_to_upsert.append(vector)

# Upload to Pinecone
print(f"Uploading {len(vectors_to_upsert)} vectors to Pinecone...")
index.upsert(vectors=vectors_to_upsert)
print("✅ Vectors uploaded to Pinecone!")
```

**What this does:**
- Connects to your Pinecone account
- Uploads all embeddings and metadata
- Now you can search this database!

---

### **STEP 4: Basic Search (BASELINE)**
**What to do:** Search for documents WITHOUT reranking (this is your baseline to compare against)

#### 4.1 - Simple search function

```python
def basic_search(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Search for documents similar to the query.
    
    Just uses embedding similarity - NOTHING fancy yet.
    """
    # 1. Convert query to embedding
    query_embedding = get_embeddings([query])[0]
    
    # 2. Search Pinecone for similar embeddings
    search_results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )
    
    # 3. Format results nicely
    results = []
    for match in search_results["matches"]:
        results.append({
            "id": match["id"],
            "score": match["score"],  # 0-1, higher = more similar
            "text": match["metadata"]["text"],
            "source_type": match["metadata"]["source_type"],
            "section": match["metadata"]["section"],
            "page": match["metadata"]["page"]
        })
    
    return results

# Test it
print("=" * 60)
print("TEST QUERY: What are the penalties for AI violations?")
print("=" * 60)
baseline_results = basic_search("What are the penalties for AI violations?", top_k=5)

for i, result in enumerate(baseline_results, 1):
    print(f"\n{i}. Score: {result['score']:.3f}")
    print(f"   Text: {result['text'][:100]}...")
    print(f"   Source: {result['source_type']} (Page {result['page']})")
```

**What this does:**
- Takes your question
- Converts it to an embedding
- Searches Pinecone for similar embeddings
- Returns top 5 results
- Shows you the score (0-1, higher = better match)

---

### **STEP 5: Add Metadata Filtering (Bonus feature)**
**What to do:** Filter results by type (e.g., only legal documents, not podcasts)

#### 5.1 - Search with filters

```python
def search_with_metadata_filter(
    query: str, 
    source_type: str = "legal",  # Only legal documents
    top_k: int = 5
) -> List[Dict[str, Any]]:
    """
    Search but ONLY return results from a specific source type.
    """
    query_embedding = get_embeddings([query])[0]
    
    # Pinecone filter - only return "legal" documents
    search_results = index.query(
        vector=query_embedding,
        top_k=top_k,
        filter={"source_type": {"$eq": source_type}},  # Filter!
        include_metadata=True
    )
    
    results = []
    for match in search_results["matches"]:
        results.append({
            "id": match["id"],
            "score": match["score"],
            "text": match["metadata"]["text"],
            "source_type": match["metadata"]["source_type"],
            "section": match["metadata"]["section"],
            "page": match["metadata"]["page"]
        })
    
    return results

# Test with filter
print("\nWith Metadata Filter (legal only):")
filtered_results = search_with_metadata_filter(
    "What are the penalties for AI violations?",
    source_type="legal",
    top_k=5
)
for result in filtered_results:
    print(f"  - {result['text'][:80]}... (Score: {result['score']:.3f})")
```

**What this does:**
- Same as basic search BUT only returns documents matching a filter
- In this case, only "legal" documents
- Useful when you have mixed content (podcasts + legal docs)

---

### **STEP 6: Implement Reranking with Cohere**
**What to do:** Use Cohere to re-score results and put the BEST ones first

#### 6.1 - Rerank using Cohere API

```python
import cohere

# Initialize Cohere client
co = cohere.ClientV2(api_key=os.getenv("COHERE_API_KEY"))

def rerank_results(
    query: str,
    results: List[Dict[str, Any]],
    top_k: int = 3
) -> List[Dict[str, Any]]:
    """
    Use Cohere to re-rank the search results.
    
    Cohere looks at:
    1. The query
    2. Each result
    3. How RELEVANT is each result to the query?
    
    Returns reranked results (best first).
    """
    # Extract just the text from results
    documents = [result["text"] for result in results]
    
    # Send to Cohere for reranking
    rerank_response = co.rerank(
        model="rerank-english-v3.0",
        query=query,
        documents=documents,
        top_n=top_k
    )
    
    # Reorder our results based on Cohere's ranking
    reranked_results = []
    for rank_result in rerank_response.results:
        original_result = results[rank_result.index]
        reranked_results.append({
            **original_result,
            "rerank_score": rank_result.relevance_score,  # New score from Cohere
            "original_rank": rank_result.index + 1  # Which position it was before
        })
    
    return reranked_results

# Test reranking
print("=" * 60)
print("STEP 1: Basic Search Results (before reranking)")
print("=" * 60)
baseline = basic_search("What are the penalties for AI violations?", top_k=5)
for i, result in enumerate(baseline, 1):
    print(f"{i}. [Score: {result['score']:.3f}] {result['text'][:70]}...")

print("\n" + "=" * 60)
print("STEP 2: Reranked Results (after Cohere reranking)")
print("=" * 60)
reranked = rerank_results(
    "What are the penalties for AI violations?",
    baseline,
    top_k=3
)
for i, result in enumerate(reranked, 1):
    print(f"{i}. [Cohere Score: {result['rerank_score']:.3f}] {result['text'][:70]}...")
    print(f"   (Was originally at position {result['original_rank']})")
```

**What this does:**
- Takes your basic search results (top 5)
- Sends them to Cohere AI
- Cohere re-evaluates how RELEVANT each is to your query
- Returns them reordered with better results first
- Shows you which results moved up/down

**Key insight:** Cohere might see that result #4 is actually most relevant, even though embedding similarity ranked it #4!

---

### **STEP 7: Complete RAG Pipeline**
**What to do:** Combine everything into one complete system

#### 7.1 - Full RAG function

```python
def rag_pipeline(
    query: str,
    use_reranking: bool = True,
    top_k_search: int = 5,
    top_k_final: int = 3
) -> Dict[str, Any]:
    """
    Complete RAG pipeline:
    1. Search for similar documents
    2. (Optional) Rerank with Cohere
    3. Return best results
    """
    
    # Step 1: Basic search
    search_results = basic_search(query, top_k=top_k_search)
    
    # Step 2: Rerank (optional)
    if use_reranking:
        final_results = rerank_results(query, search_results, top_k=top_k_final)
        method = "WITH Reranking"
    else:
        final_results = search_results[:top_k_final]
        method = "WITHOUT Reranking"
    
    return {
        "query": query,
        "method": method,
        "results": final_results,
        "count": len(final_results)
    }

# Test the full pipeline
print("=" * 70)
print("FULL RAG PIPELINE TEST")
print("=" * 70)

# Without reranking
print("\n🔍 WITHOUT Reranking:")
result_without = rag_pipeline(
    "What are the penalties for AI violations?",
    use_reranking=False,
    top_k_final=3
)
for i, doc in enumerate(result_without["results"], 1):
    print(f"{i}. {doc['text'][:80]}...")

# With reranking
print("\n🚀 WITH Reranking:")
result_with = rag_pipeline(
    "What are the penalties for AI violations?",
    use_reranking=True,
    top_k_final=3
)
for i, doc in enumerate(result_with["results"], 1):
    score = doc.get('rerank_score', doc['score'])
    print(f"{i}. [Score: {score:.3f}] {doc['text'][:80]}...")
```

**What this does:**
- Creates one function that does EVERYTHING
- You pass in a question
- It searches, optionally reranks, and returns top results
- Easy to toggle reranking on/off for comparison

---

### **STEP 8: Evaluation & Comparison**
**What to do:** Compare results WITH and WITHOUT reranking to see if it helps

#### 8.1 - Compare two approaches

```python
def evaluate_rag(test_queries: List[str]) -> Dict[str, Any]:
    """
    Test the RAG system with multiple questions.
    Compare with and without reranking.
    """
    
    comparison_table = []
    
    for query in test_queries:
        # Get results both ways
        without_rerank = rag_pipeline(query, use_reranking=False, top_k_final=3)
        with_rerank = rag_pipeline(query, use_reranking=True, top_k_final=3)
        
        comparison_table.append({
            "query": query,
            "without_reranking": without_rerank["results"],
            "with_reranking": with_rerank["results"]
        })
    
    return comparison_table

# Test queries relevant to AI law
test_queries = [
    "What are the penalties for AI violations?",
    "What is high-risk AI according to EU AI Act?",
    "How should AI systems be audited?",
    "What transparency requirements exist for AI?",
    "What are the prohibited AI practices?"
]

print("=" * 70)
print("EVALUATION: Testing RAG System")
print("=" * 70)

results = evaluate_rag(test_queries)

for comparison in results:
    print(f"\n📌 Query: {comparison['query']}")
    print("-" * 70)
    
    print("\nWithout Reranking (top 3):")
    for i, doc in enumerate(comparison['without_reranking'], 1):
        print(f"  {i}. {doc['text'][:60]}...")
    
    print("\nWith Reranking (top 3):")
    for i, doc in enumerate(comparison['with_reranking'], 1):
        score = doc.get('rerank_score', doc['score'])
        print(f"  {i}. {doc['text'][:60]}... (score: {score:.3f})")
    
    print()
```

**What this does:**
- Runs 5 different questions through the system
- Shows results with and without reranking
- You can manually check which ones are better
- This helps you see when reranking helps most

---

## 📊 **What You Should Observe**

### **Without Reranking:**
- Results are ordered by embedding similarity only
- Sometimes less relevant results appear high
- All results might be somewhat relevant, but not in best order

### **With Reranking:**
- Results are reordered by actual relevance
- Highly relevant results move to top
- More "human-like" ordering
- Usually 2-3 best results are clearly better

---

## 💡 **When to Use Reranking**

**Use reranking WHEN:**
- ✅ Legal/official documents (needs precision)
- ✅ Long documents with many similar sections
- ✅ Diverse document types (podcasts + legal)
- ✅ High stakes (wrong answer = big problem)

**DON'T use reranking WHEN:**
- ❌ Simple questions with clear answers
- ❌ Budget is tight (reranking costs money)
- ❌ Speed is critical (adds 1-2 second latency)
- ❌ You have only a few documents anyway

---

## 🎓 **Key Concepts Explained Simply**

### **Embeddings**
- Text → Numbers that computers understand
- Similar text → similar numbers
- "EU AI Act" and "European AI legislation" → close embeddings

### **Vector Database (Pinecone)**
- Stores millions of embeddings
- Finds similar embeddings FAST
- Like a super-smart search index

### **Reranking**
- Takes top search results
- Re-evaluates them more carefully
- Puts best ones first
- Like a quality control step

### **Metadata**
- Extra info about each document
- Source type, page number, section
- Helps filter (e.g., "legal documents only")

---

## ✅ **Checklist: What To Submit**

When you're done, you need:

1. **Complete notebook** with:
   - ✅ Data loading
   - ✅ Embedding generation
   - ✅ Pinecone upload
   - ✅ Basic search
   - ✅ Metadata filtering
   - ✅ Cohere reranking
   - ✅ Complete RAG pipeline
   - ✅ Evaluation results

2. **Example outputs** showing:
   - Results WITHOUT reranking
   - Results WITH reranking
   - Comparison table

3. **lab_summary.md** with 1 short paragraph:
   - When did reranking help the most?
   - When would you NOT use it?
   - What did you learn?

4. **README.md** with:
   - How to run your code
   - What files do what

---

## 🚨 **Common Issues & Solutions**

### **Issue: "API key not found"**
**Solution:** Make sure `.env` file exists and has correct format:
```
OPENAI_API_KEY=sk-...
COHERE_API_KEY=...
PINECONE_API_KEY=...
```

### **Issue: "Pinecone index not found"**
**Solution:** Create the index first in Pinecone console, or let the code create it

### **Issue: "Embeddings are slow"**
**Solution:** Use `text-embedding-3-small` (faster) instead of large model

### **Issue: "Reranking gives bad results"**
**Solution:** Try different queries or adjust top_k (maybe return more results before reranking)

---

## 📚 **Summary**

You now have a complete RAG system that:
1. ✅ Loads documents
2. ✅ Converts them to embeddings
3. ✅ Stores in vector database
4. ✅ Searches with similarity
5. ✅ Re-ranks with Cohere
6. ✅ Returns best results
7. ✅ Compares with/without reranking




