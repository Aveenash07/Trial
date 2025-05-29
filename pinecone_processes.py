from pinecone import Pinecone
from utils import split_text, create_embedding
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()


client = OpenAI(api_key=os.getenv("OPEN_API_KEY"))


class RAGBot:
    def __init__(self):
        # Initialize Pinecone client
        self.pc = Pinecone(
            api_key=os.getenv("PINECONE_API_KEY"),
            environment=os.getenv("PINECONE_ENVIRONMENT"),
        )
        self.index = self.pc.Index(name="trial-index")

    def store_document(self, text):
        chunks = split_text(text)
        for i, chunk in enumerate(chunks):
            embedding = create_embedding(chunk)
            self.index.upsert(vectors=[(f"chunk_{i}", embedding, {"text": chunk})])
        return "Document stored successfully"

    def answer_question(self, query):
        query_embedding = create_embedding(query)
        result = self.index.query(
            vector=query_embedding, top_k=3, include_metadata=True
        )
        contexts = [match["metadata"]["text"] for match in result["matches"]]

        separator = "\n---\n\n"
        context_str = separator.join(contexts)
        prompt = f"""
        Answer the question based on the context below.
        
        Context:
        {context_str}
        
        Question: {query}
        Answer:
        """

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,  # Corrected temperature (0 to 2 range, 0.7 is reasonable)
        )
        return response.choices[0].message.content
