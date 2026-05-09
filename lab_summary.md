# Lab Summary

Reranking helped most when queries were **[describe the type of query — e.g. specific legal obligations, definitions]**, because the baseline cosine similarity often surfaced chunks that *mentioned* the right keywords but lacked the actual answer. For example, for the query "What are the obligations for high-risk AI systems?", the baseline's top result was about [X], while the Cross-Encoder correctly surfaced page [Y] which directly described [Z].

I would recommend using a reranker whenever **retrieval precision is critical** — particularly in legal, compliance, or medical domains where a wrong chunk can produce a confidently wrong answer. The **Cross-Encoder** is the best choice for local/offline use with no API cost, while **Cohere Rerank** is preferable in production where latency matters and a small per-query cost is acceptable. LLM-based scoring is useful when you need explainability but is too slow for interactive applications.
