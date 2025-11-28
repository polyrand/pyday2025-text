import argparse
import json
import os
import urllib.request


def get_embedding(text: str) -> list[float]:
    """
    Get embedding for text using OpenAI API.

    Args:
        text: The text to embed

    Returns:
        List of floats representing the embedding
    """

    # This is the only supported model. The API key is limited and won't allow
    # other embedding models except this one.
    EMBEDDING_MODEL = "text-embedding-3-small"
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set")

    url = "https://api.openai.com/v1/embeddings"

    data = json.dumps({"input": text, "model": EMBEDDING_MODEL}).encode("utf-8")

    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}

    req = urllib.request.Request(url, data=data, headers=headers, method="POST")

    with urllib.request.urlopen(req) as response:
        result = json.loads(response.read().decode("utf-8"))

    return result["data"][0]["embedding"]


def main():
    parser = argparse.ArgumentParser(description="Get embeddings from OpenAI API")
    parser.add_argument("--text", required=True, help="Text to embed")
    parser.add_argument("--output", type=str, help="Output file to save the embedding")
    args = parser.parse_args()

    embedding = get_embedding(args.text)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(embedding, f)
    else:
        print(json.dumps(embedding))


if __name__ == "__main__":
    main()
