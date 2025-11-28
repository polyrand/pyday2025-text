"""Query semantic search using embeddings and cosine similarity."""

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path


def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """Calculate cosine similarity between two vectors.

    Cosine similarity measures the cosine of the angle between two vectors.
    It is widely used in text similarity because it measures orientation
    (direction) rather than magnitude, making it ideal for comparing
    documents of different lengths.

    The formula is:
        cosine_similarity(A, B) = (A · B) / (||A|| * ||B||)

    Where:
        - A · B is the dot product of vectors A and B
        - ||A|| is the magnitude (Euclidean norm) of vector A
        - ||B|| is the magnitude (Euclidean norm) of vector B

    The result ranges from -1 to 1:
        -  1 means vectors point in the same direction (identical)
        -  0 means vectors are orthogonal (unrelated)
        - -1 means vectors point in opposite directions

    Args:
        vec1: First vector as a list of floats.
        vec2: Second vector as a list of floats.

    Returns:
        Cosine similarity score between -1 and 1.
    """
    assert len(vec1) == len(vec2), "Vectors must be the same length"
    assert len(vec1) > 0, "Vectors must not be empty"
    # Step 1: Calculate the dot product (A · B)
    # The dot product is the sum of element-wise multiplications.
    # It measures how much two vectors "agree" in direction.
    # Formula: A · B = a1*b1 + a2*b2 + ... + an*bn
    dot_product = sum(a * b for a, b in zip(vec1, vec2))

    # Alternative:
    # total = 0.0
    # for a, b in zip(vec1, vec2):
    #     total += a * b
    # dot_product = total

    # Step 2: Calculate the magnitude (Euclidean norm) of each vector
    # The magnitude is the "length" of the vector in n-dimensional space.
    # Formula: ||A|| = sqrt(a1² + a2² + ... + an²)
    # We sum the squares of all elements, then take the square root.
    magnitude1 = math.sqrt(sum(a * a for a in vec1))
    magnitude2 = math.sqrt(sum(b * b for b in vec2))

    # Step 3: Handle edge case where a vector has zero magnitude
    # A zero vector has no direction, so similarity is undefined.
    # We return 0.0 to indicate no similarity in this case.
    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0

    # Step 4: Compute cosine similarity by dividing dot product by magnitudes
    # This normalizes the result to be independent of vector lengths.
    return dot_product / (magnitude1 * magnitude2)


def get_query_embedding(query: str) -> list[float]:
    """Generate embedding for a query using semantic_oai.py subprocess.

    Args:
        query: The query text to embed.

    Returns:
        List of floats representing the embedding.
    """
    result = subprocess.run(
        [sys.executable, "semantic_oai.py", "--text", query],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout)


def query_semantic(query: str) -> dict[str, dict]:
    """Query documents and return similarity scores with document text.

    This function performs semantic search by:
    1. Converting the query text into a numerical embedding vector
    2. Loading pre-computed embeddings for all documents
    3. Comparing the query embedding against each document embedding
    4. Returning similarity scores and document text ranked by relevance

    Semantic search differs from keyword search because it understands
    meaning and context, not just exact word matches. Two sentences
    can be semantically similar even if they share no common words.

    Args:
        query: The query text to search for.

    Returns:
        Dictionary mapping document_id (filename) to a dict containing
        'score' (cosine similarity) and 'text' (document content).
        Higher scores indicate more relevant documents.
    """
    # Step 1: Generate embedding for the query text
    # The embedding is a high-dimensional vector (e.g., 768 dimensions)
    # that captures the semantic meaning of the query.
    query_embedding = get_query_embedding(query)

    # Step 2: Set up the embeddings and data directory paths
    # Pre-computed embeddings are stored as JSON files in embeddings directory.
    # Original documents are stored as text files in data directory.
    embeddings_dir = Path("embeddings")
    data_dir = Path("data")
    results = {}

    # Step 3: Iterate through all document embeddings
    # Each file is named like "1_embedding.json", "2_embedding.json", etc.
    for embedding_file in embeddings_dir.glob("*_embedding.json"):
        # Step 3a: Extract the document ID from the filename
        # e.g., "1_embedding.json" -> "1"
        doc_id = embedding_file.stem.replace("_embedding", "")

        # Step 3b: Load the pre-computed document embedding from JSON
        with open(embedding_file, "r", encoding="utf-8") as f:
            doc_embedding = json.load(f)

        # Step 3c: Calculate cosine similarity between query and document
        # This measures how semantically similar the query is to the document.
        similarity = cosine_similarity(query_embedding, doc_embedding)

        # Step 3d: Load the original document text
        doc_file = data_dir / f"{doc_id}.txt"
        doc_text = ""
        if doc_file.exists():
            with open(doc_file, "r", encoding="utf-8") as f:
                doc_text = f.read()

        # Step 3e: Store the similarity score and text mapped to the document ID
        results[doc_id] = {"score": similarity, "text": doc_text}

    # Step 4: Return all results with scores and text
    # The caller can sort these to find the most relevant documents.
    return results


def main():
    parser = argparse.ArgumentParser(description="Semantic search query")
    parser.add_argument("--query", required=True, help="Query text to search for")
    parser.add_argument(
        "--top-k", type=int, default=5, help="Number of top results to return"
    )
    args = parser.parse_args()

    results = query_semantic(args.query)
    # Sort results by similarity score in descending order
    sorted_results = dict(
        sorted(results.items(), key=lambda item: item[1]["score"], reverse=True)
    )
    # Return only the top-k results
    top_results = {k: sorted_results[k] for k in list(sorted_results)[: args.top_k]}
    print(json.dumps(top_results, indent=2))


if __name__ == "__main__":
    main()
