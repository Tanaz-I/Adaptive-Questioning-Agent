NLP Architecture

Input (Text / Images)  
        ↓  
   OCR (if needed)  
        ↓  
Topic Extraction → Concept Graph  
        ↓  
Chunking + Merging  
        ↓  
BM25 + RRF Retrieval  
        ↓  
Question Generation Engine  
        ↓  
Student Simulation  
        ↓  
Scoring + Feedback  
        ↓  
Material Recommendation  
        ↓  
Frontend Display


Retrieval System Comparison(retrieval_engine.py)
❌ BEFORE (Baseline System)

Approach:

Single query using topic name
Semantic similarity search
Strict metadata filtering
⚠️ Limitations
Limited coverage
Only one form of query used
Misses related concepts
Redundant retrieval
Similar chunks retrieved multiple times
Poor support for reasoning
Mostly definitions retrieved
Weak inferential questions
Brittle filtering
If metadata mismatched → no results
✅ AFTER (Improved System)

Approach:

Multi-query expansion
Hybrid scoring (semantic + metadata)
Diversity-based selection
Fallback retrieval
🔥 Key Improvements
1. Query Expansion

Multiple query variations generated:

explanation
example
definition
applications
difficulty-aware
question-type-aware
2. Multi-Query Retrieval

Each query independently retrieves relevant chunks → combined

3. Fallback Mechanism

If strict filtering fails:

system retrieves without filter
4. Metadata-Aware Scoring ⭐

Chunks boosted based on usefulness:

Type	Boost
Explanation	+0.05
Example	+0.03
5. Diversity (MMR-style)

Ensures:

different subtopics
no repetition

