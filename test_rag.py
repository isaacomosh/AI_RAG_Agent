from pdf_processor import extract_text_from_pdf
from rag import rerank_chunks   
from rag import (
    chunk_text,
    store_chunks,
    retrieve_chunks
)

from gemini_client import generate_answer


# 1. Extract PDF text
pages = extract_text_from_pdf("The Magic of Kyoto.pdf")


# 2. Split text into chunks
chunks = chunk_text(pages)

print("Total chunks:", len(chunks))


# 3. Store chunks
store_chunks(chunks)


# 4. Ask a question
question = input("\nAsk something about your PDF: ")


# 5. Retrieve relevant chunks
results = retrieve_chunks(question)



#Retrieve 10 candidate chunks
results=retrieve_chunks(question,n_results=10)

reranked_results=rerank_chunks(
    question,results,top_k=3
)



# 6. Combine retrieved chunks into context
context = ""

for chunk  in reranked_results:
    context+=f"""
        Page: {chunk['page']}
        Content:
        {chunk['text']}
    """
    

# 7. Generate Gemini response
answer = generate_answer(question, context)


# 8. Display answer
print("\nAI ANSWER:")
print(answer)