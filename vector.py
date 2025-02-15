from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

def read_document(file_path):
    """Reads the content of a document."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read().strip().lower()  # Lowercasing for uniformity
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None

def check_document_similarity(file1, file2, threshold=0.7):
    """Computes cosine similarity between two text documents using BERT embeddings."""
    
    # Read documents
    doc1 = read_document(file1)
    doc2 = read_document(file2)

    if doc1 is None or doc2 is None:
        return

    # Load a pre-trained BERT-based sentence transformer model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Generate embeddings for both documents
    embeddings = model.encode([doc1, doc2])

    # Compute cosine similarity
    similarity_score = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]

    # Print similarity score
    print(f"Cosine Similarity: {similarity_score:.4f}")

    # Check if documents are similar
    if similarity_score >= threshold:
        print("✅ Documents are similar.")
    else:
        print("❌ Documents are not similar.")

# Example usage
file1 = input("Enter the path of the first document: ")
file2 = input("Enter the path of the second document: ")

check_document_similarity(file1, file2)
