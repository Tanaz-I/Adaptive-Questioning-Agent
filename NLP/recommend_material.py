import chromadb
import pandas as pd
import numpy as np

from chromadb.config import Settings
from collections import defaultdict
from sentence_transformers import SentenceTransformer

# CONFIG
CHROMA_DB_DIR   = "./chroma_db"
COLLECTION_NAME = "rag_kb"
CSV_PATH        = "./Online_Courses.csv"
TOP_N_COURSES   = 5


# load embedding model once
_embed_model = None
def _get_embed_model():

    global _embed_model
    if _embed_model is None:
        print("Loading embedding model...")
        _embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embed_model

# =====================================================
# MATERIAL RECOMMENDATION (FILES)
# =====================================================

def get_weak_topic_material(weak_topics: list[str]) -> list[str]:

    """
    returns:
        [
            "lecture1.pdf",
            "memory.pdf",
            "constructors.pdf"
        ]
    """

    if not weak_topics:
        return []

    client = chromadb.PersistentClient(
        path = CHROMA_DB_DIR,
        settings = Settings(anonymized_telemetry=False)
    )

    collection = client.get_collection(COLLECTION_NAME)
    data = collection.get(include = ["metadatas"])

    metas = data["metadatas"]
    weak_set = {
        topic.strip().lower()
        for topic in weak_topics
    }
    
    unique_files = set()
    
    for meta in metas:

        topic = (meta.get("topic") or "").strip().lower()
        
        subtopic = (
            meta.get("subtopic")
            or ""
        ).strip().lower()


        if topic in weak_set or subtopic in weak_set:
            file_name = meta.get("file_name")
            if file_name:
                unique_files.add(file_name)

    return sorted(list(unique_files))



# =====================================================
# COURSE RECOMMENDATION
# =====================================================

_course_df         = None
_course_embeddings = None


def _load_courses():

    global _course_df
    global _course_embeddings

    if _course_df is not None:
        return _course_df, _course_embeddings

    print("Loading course dataset...")

    df = pd.read_csv(CSV_PATH)
    english_mask = (df["Language"].isna() | df["Language"].str.lower().str.contains("english",na = False))

    df = df[english_mask].reset_index(drop = True)
    df = df.dropna(subset = ["Title","URL"]).reset_index(drop = True)
    df = df.drop_duplicates(subset = ["URL"]).reset_index(drop = True)
    
    def build_text(row):

        parts = [ str(row["Title"])]
        
        if pd.notna(row.get("Short Intro")):
            parts.append(str(row["Short Intro"])[:300])

        if pd.notna(row.get("Skills")):
            parts.append(str(row["Skills"])[:200])
            
        return " | ".join(parts)

    texts = df.apply(build_text, axis = 1).tolist()
    model = _get_embed_model()
    embeddings = model.encode(texts,normalize_embeddings = True)
    
    _course_df         = df
    _course_embeddings = embeddings
    return _course_df, _course_embeddings

def recommend_courses(weak_topics, top_n = TOP_N_COURSES):
    if not weak_topics:
        return []
    df, embeddings = _load_courses()
    model = _get_embed_model()
    queries = [" ".join(weak_topics)] + weak_topics
    query_embeddings = model.encode(queries,normalize_embeddings = True)
    scores_matrix = np.dot( query_embeddings, embeddings.T)
    max_scores = scores_matrix.max(axis = 0)

    top_indices = np.argsort( max_scores)[::-1][: top_n * 3 ]
    seen_urls = set()
    results   = []
    for idx in top_indices:
        url = str(df.iloc[idx]["URL"]).strip()
        if url in seen_urls:
            continue
        seen_urls.add(url)
        results.append({
            "title":
                str(
                    df.iloc[idx]["Title"]
                ),

            "url":
                url
        })


        if len(results) >= top_n:
            break
        
    return results