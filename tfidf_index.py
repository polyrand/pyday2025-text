"""
TF-IDF Index Builder

A simple implementation of TF-IDF (Term Frequency-Inverse Document Frequency)
for building a search index from text documents.
"""

import json
import math
from collections import Counter
from pathlib import Path

"""
TASKS:

1. [DONE] Build mapping of docid -> filepath
2. Build mapping of docid -> tokens
3. Calculate document frequency (DF) for each term
4. Calculate IDF for each term
5. Calculate TF-IDF for each term in each document
6. Implement saving/loading index to/from JSON
7. Implement search function that ranks by TF-IDF scores
8. Improve tokenizer?

"""


def tokenize(text: str) -> list[str]:
    """
    Tokenize a string
    """
    tokens = [word.casefold().strip() for word in text.split() if word]
    return tokens


def tokenize_file(path: Path) -> list[str]:
    """
    Tokenize the contents of a file
    """
    content = path.read_text(encoding="utf-8")
    return tokenize(content)


def build_tfidf_index(data_folder: Path) -> dict:
    """
    Build a TF-IDF index from all .txt files in the given folder.

    ============================================================================
    TF-IDF EXPLANATION
    ============================================================================

    TF-IDF stands for Term Frequency-Inverse Document Frequency. It is a
    numerical statistic used to reflect how important a word is to a document
    in a collection (corpus). It's commonly used in information retrieval and
    text mining.

    The intuition behind TF-IDF:
    - Words that appear frequently in a document are important TO THAT DOCUMENT
    - Words that appear in many documents are LESS DISTINCTIVE (e.g., "the", "is")
    - The best keywords are those that appear often in one document but rarely
      across all documents

    ============================================================================
    THE FORMULA
    ============================================================================

    TF-IDF(term, document, corpus) = TF(term, document) × IDF(term, corpus)

    Where:

    1. TF (Term Frequency) - How often a term appears in a document

       TF(t, d) = (Number of times term t appears in document d)
                  ─────────────────────────────────────────────────
                  (Total number of terms in document d)

       Example: If "machine" appears 5 times in a 100-word document:
                TF("machine", doc) = 5/100 = 0.05

    2. IDF (Inverse Document Frequency) - How rare/common a term is across corpus

       IDF(t, D) = log( N / df(t) )

       Where:
         - N = Total number of documents in corpus
         - df(t) = Number of documents containing term t

       Example: If we have 60 documents and "machine" appears in 10 of them:
                IDF("machine") = log(60/10) = log(6) ≈ 1.79

       Note: We add 1 to denominator to avoid division by zero:
             IDF(t, D) = log( N / (df(t) + 1) ) + 1

    3. Final TF-IDF Score

       TF-IDF(t, d, D) = TF(t, d) × IDF(t, D)

       Example: TF-IDF("machine", doc, corpus) = 0.05 × 1.79 = 0.0895

    ============================================================================
    STEP-BY-STEP PROCESS
    ============================================================================

    Step 1: LOAD DOCUMENTS
            Read all text files from the data folder into memory.
            Store as {filename: content} dictionary.

    Step 2: TOKENIZE
            Convert each document into a list of words (tokens).
            - Convert to lowercase for consistency
            - Remove punctuation and special characters
            - Split on whitespace

    Step 3: CALCULATE DOCUMENT FREQUENCY (DF)
            Count how many documents each term appears in.
            df["machine"] = 10 means "machine" is in 10 documents.

    Step 4: CALCULATE IDF FOR EACH TERM
            IDF measures term rarity across the corpus.
            Rare terms get higher IDF scores.
            Common terms (like "the") get lower IDF scores.

    Step 5: CALCULATE TF-IDF FOR EACH TERM IN EACH DOCUMENT
            For every (term, document) pair:
            - Calculate TF (term frequency in that document)
            - Multiply by IDF (from step 4)
            - Store the result

    ============================================================================
    RETURN VALUE STRUCTURE
    ============================================================================

    Returns a dictionary with:
    {
        "docid2path": {
            1: "/path/to/data/1.txt",
            2: "/path/to/data/2.txt",
            ...
        },
        "idf": {
            "machine": 1.79,
            "learning": 2.01,
            ...
        },
        "tfidf": {
            1: {
                "machine": 0.0895,
                "learning": 0.0712,
                ...
            },
            2: {...},
            ...
        },
        "doc_count": 60
    }

    Args:
        data_folder: Path to folder containing .txt files

    Returns:
        Dictionary containing the TF-IDF index
    """

    all_documents_paths = list(Path(data_folder).glob("*.txt"))

    # =========================================================================
    # STEP 1: Create mapping of document ID to file path
    # This will be the base dictionary to use for tokenization and later
    # retrieval of the full document text.
    # =========================================================================
    docid2path: dict[int, Path] = {}
    NUMBER_OF_DOCUMENTS = 0
    for filepath in all_documents_paths:
        doc_id = int(filepath.stem)  # Get the file name without extension
        docid2path[doc_id] = filepath
        NUMBER_OF_DOCUMENTS += 1

    # =========================================================================
    # STEP 2: Tokenize documents.
    # Build a mapping of:
    #     document_id -> [list, of, tokens]
    # We will use this to calculate TF and DF
    # =========================================================================
    tokenized_docs: dict[int, list[str]] = {}
    for doc_id, path in docid2path.items():
        tokenized_docs[doc_id] = tokenize_file(path)

    # =========================================================================
    # STEP 3: Calculate Document Frequency (DF)
    # Build a mapping of:
    #     term -> number of documents containing the term
    # =========================================================================
    document_frequency: Counter[str] = Counter()
    for _doc_id, tokens in tokenized_docs.items():
        unique_terms = set(tokens)
        for term in unique_terms:
            document_frequency[term] += 1

    # =========================================================================
    # STEP 4: Calculate IDF for each term
    #
    # N: Total number of documents
    # df: Document frequency (number of documents containing the term)
    # t: The term
    # D: The document corpus
    #
    # Formula: IDF(t, D) = log( N / (df(t) + 1) ) + 1
    #
    # WHY LOGARITHM?
    # The log serves two purposes:
    #
    # 1. DAMPENING: Without log, IDF would be a simple ratio (N/df).
    #    This creates extreme values:
    #    - Term in 1 of 60 docs: without log = 60, with log ≈ 4.1
    #    - Term in 30 of 60 docs: without log = 2, with log ≈ 0.7
    #    The log compresses the scale, preventing rare terms from
    #    completely dominating the score.
    #
    # 2. MATCHING HUMAN PERCEPTION: Relevance doesn't scale linearly
    #    with rarity. A term in 1 document isn't 10x more important
    #    than one in 10 documents—it's *somewhat* more important.
    #    Log follows the Weber-Fechner law: humans perceive differences
    #    on a logarithmic scale (like decibels or earthquake magnitudes).
    # =========================================================================
    idf = {}
    for term, df in document_frequency.items():
        # IDF formula with smoothing: log(N / (df + 1)) + 1
        idf[term] = math.log(NUMBER_OF_DOCUMENTS / (df + 1)) + 1

    # =========================================================================
    # STEP 5: Calculate TF-IDF for each term in each document
    # =========================================================================
    tfidf: dict[int, dict[str, float]] = {}
    for doc_id, tokens in tokenized_docs.items():
        term_counts = Counter(tokens)
        total_terms = len(tokens)

        tfidf[doc_id] = {}
        # store scores of each term
        for term, count in term_counts.items():
            term_frequency = count / total_terms  # TF
            inverse_document_frequency = idf[term]  # IDF
            tfidf[doc_id][term] = term_frequency * inverse_document_frequency

    # =========================================================================
    # Return the complete index
    # =========================================================================
    return {
        "docid2path": docid2path,
        "idf": idf,
        "tfidf": tfidf,
        "doc_count": NUMBER_OF_DOCUMENTS,
    }


def search(index: dict, terms: list[str]) -> list[tuple[int, float]]:
    """
    Search for documents containing multiple terms, ranked by summed TF-IDF scores.

    ============================================================================
    MULTI-TERM SEARCH EXPLANATION
    ============================================================================

    This search accepts multiple terms and returns documents ranked by the
    sum of TF-IDF scores for all matching terms.

    Process:
    1. Lowercase all query terms
    2. For each document, sum the TF-IDF scores for all matching terms
    3. Add (document, total_score) to results if any term matches
    4. Sort by score descending (highest relevance first)

    Why summing scores works for ranking:
    - Documents matching more terms get higher scores
    - Documents where terms are more important (higher TF-IDF) rank higher
    - Combines relevance across all search terms

    Args:
        index: The TF-IDF index from build_tfidf_index()
        terms: List of search terms

    Returns:
        List of (filename, score) tuples sorted by relevance
    """
    terms = [t.lower() for t in terms]
    doc_scores: dict[int, float] = {}

    # Sum scores for each term in each document
    for doc_id, doc_tfidf in index["tfidf"].items():
        total_score = 0.0
        for term in terms:
            # weight = 1.0
            # if term_has_weight(term):
            #     weight = get_term_weight(term)
            if term in doc_tfidf:
                score = doc_tfidf[term]
                # score *= weight
                total_score += score
        if total_score > 0:
            doc_scores[doc_id] = total_score

    # Sort by score descending (most relevant first)
    results = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="TF-IDF Index Builder and Search Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tfidf_index.py --build index.json              Build the index and save to JSON
  python tfidf_index.py --index-file index.json --query "machine learning"  Search using saved index
  python tfidf_index.py --index-file index.json --query "python"            Single-term search
        """,
    )
    parser.add_argument(
        "--build",
        type=str,
        metavar="OUTPUT_PATH",
        help="Build the TF-IDF index and save to specified JSON file path",
    )
    parser.add_argument(
        "--index-file",
        type=str,
        metavar="INDEX_PATH",
        help="Path to a pre-built JSON index file (required for --query)",
    )
    parser.add_argument(
        "--query",
        type=str,
        help="Search query (single or multiple words). Requires --index-file",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="Number of results to show (default: 10)",
    )

    args = parser.parse_args()

    if args.query and not args.index_file:
        parser.error("--query requires --index-file")

    if args.build:
        data_path = Path(__file__).parent / "data"
        index = build_tfidf_index(data_path)
        # Save index to JSON file
        output_path = args.build
        # Convert Path objects to strings for JSON serialization
        serializable_index = {
            "docid2path": {k: str(v) for k, v in index["docid2path"].items()},
            "idf": index["idf"],
            "tfidf": dict(index["tfidf"]),
            "doc_count": index["doc_count"],
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(serializable_index, f, indent=2)
        print(f"Index saved to {output_path}")
        print(f"Indexed {index['doc_count']} documents")
        print(f"Vocabulary size: {len(index['idf'])} terms")

    if args.query:
        # Load index from JSON file
        with open(args.index_file, "r", encoding="utf-8") as f:
            loaded_index = json.load(f)
        # Convert string keys back to int for tfidf
        index = {
            "docid2path": {int(k): v for k, v in loaded_index["docid2path"].items()},
            "idf": loaded_index["idf"],
            "tfidf": {int(k): v for k, v in loaded_index["tfidf"].items()},
            "doc_count": loaded_index["doc_count"],
        }

        print(f"Searching for: '{args.query}'")
        print("-" * 50)

        # Tokenize the query to get multiple terms
        query_terms = tokenize(args.query)
        results = search(index, query_terms)

        if not results:
            print("No results found.")
        else:
            # Show top N results
            for i, (doc_id, score) in enumerate(results[: args.top], 1):
                doc_path = index["docid2path"][doc_id]
                print(f"{i:2}. {doc_id:<15} (score: {score:.4f})")
                print(f"    Path: {doc_path}")

            if len(results) > args.top:
                print(f"\n... and {len(results) - args.top} more results")
