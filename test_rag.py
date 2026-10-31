from pdf_processor import extract_text_from_pdf

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


# 6. Combine retrieved chunks into context
context = ""

for document, metadata in zip(
    results["documents"][0],
    results["metadatas"][0]
):

    context += f"""
    Page {metadata['page']}:

    {document}

    """


# 7. Generate Gemini response
answer = generate_answer(question, context)


# 8. Display answer
print("\nAI ANSWER:")
print(answer)