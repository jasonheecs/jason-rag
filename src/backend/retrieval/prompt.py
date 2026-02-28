"""Prompt building module for generating LLM responses."""
from typing import Dict, List

from openai import OpenAI


class PromptBuilder:
    """Builds prompts and generates answers using LLM."""

    def __init__(self, openai_api_key: str, model: str = "gpt-4o-mini"):
        self.client = OpenAI(api_key=openai_api_key)
        self.model = model

    def build_context(self, documents: List[Dict]) -> str:
        """Build context string from retrieved documents."""
        context_parts = []

        for i, doc in enumerate(documents, 1):
            context_parts.append(
                f"[Source {i}] {doc['title']} ({doc['source']})\n"
                f"{doc['content']}\n"
            )

        return "\n".join(context_parts)

    def generate_answer(self, question: str, context: str) -> str:
        """Generate answer using OpenAI API with retrieved context."""
        prompt = f"""You are an AI assistant answering questions based on Jason's
writing and profile. The following is what you know about him, use it to answer the question.
If the answer is not in what you know about him, say that you do not know him well enough
to answer the question.

What I know about Jason:
{context}

Question: {question}

Answer:"""

        system_message = (
            "You are a helpful assistant that answers questions"
            "based on what you know about Jason."
        )
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )

        return response.choices[0].message.content

    def answer_question(self, question: str, retrieved_docs: List[Dict]) -> Dict:
        """
        Answer a question using retrieved documents.

        Args:
            question: User's question
            retrieved_docs: Documents retrieved from vector search

        Returns:
            Dict with 'answer' and 'sources'
        """
        # Build context from retrieved documents
        context = self.build_context(retrieved_docs)

        # Generate answer
        answer = self.generate_answer(question, context)

        return {
            'answer': answer,
            'sources': retrieved_docs
        }
