from sentence_transformers import SentenceTransformer
import chromadb
from sentence_transformers import CrossEncoder

#initialize ChromaDB
client=chromadb.PersistentClient(path="./chroma_db")
collection=client.get_or_create_collection(name="pdf_documents")
embedding_model=SentenceTransformer("all-MiniLM-L6-v2")
def chunk_text(pages, chunk_size=500, overlap=100):
    chunks=[]
    for page in pages:
        text=page["text"]
        start=0
        while start<len(text):
            end= start + chunk_size
            chunk=text[start:end]

            chunks.append({
                "text":chunk,"page":page["page"]
            })

            start += chunk_size-overlap
    return chunks

def generate_embeddings(chunks):
    texts=[chunk["text"]for chunk in chunks]
    embeddings=embedding_model.encode(texts)
    return embeddings

def store_chunks(chunks):
    embeddings=generate_embeddings(chunks)
    for index, chunk in enumerate(chunks):
        collection.add(
            ids=[str(index)],
             documents=[chunk["text"]]  
             ,
             embeddings=[embeddings[index].tolist()],
             metadatas=[{
                 "page":chunk["page"]
             }]
        
        )
    print("CHunks successfully stored!")


#Semantic retrieval
#where we use RAG retrieval process suppose the user asks: What is machine learning? We convert that question into an embedding and search ChromaDB for similar vectors
def retrieve_chunks(question, n_results=3):

    question_embedding = embedding_model.encode(
        [question]
    )

    results = collection.query(
        query_embeddings=[
            question_embedding[0].tolist()
        ],
        n_results=n_results
    )

    return results