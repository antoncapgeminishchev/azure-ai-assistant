import json
import numpy as np

from typing import List
from .AnsweringToolBase import AnsweringToolBase

from langchain.chains.llm import LLMChain
from langchain_core.documents import Document
from langchain.prompts import PromptTemplate
from langchain_community.callbacks import get_openai_callback
from langchain_community.vectorstores.azuresearch import (
    FIELDS_CONTENT_VECTOR,
    FIELDS_CONTENT, FIELDS_METADATA)

from ..helpers.AzureSearchHelper import AzureSearchHelper
from ..helpers.ConfigHelper import ConfigHelper
from ..helpers.LLMHelper import LLMHelper
from ..helpers.EnvHelper import EnvHelper
from ..common.Answer import Answer
from ..common.SourceDocument import SourceDocument
from azure.search.documents.models import VectorizedQuery

class QuestionAnswerTool(AnsweringToolBase):
    def __init__(self) -> None:
        env_helper = EnvHelper()
        self.name = "QuestionAnswer"
        self.vector_store = AzureSearchHelper().get_vector_store()
        self.verbose = True
        self.search_k_nearest_neighbors = env_helper.search_k_nearest_neighbors
        self.search_score_threshold = env_helper.search_score_threshold

    def answer_question(self, question: str, chat_history: List[dict], **kwargs: dict):
        config = ConfigHelper.get_active_config_or_default()
        answering_prompt = PromptTemplate(
            template=config.prompts.answering_prompt,
            input_variables=["question", "sources"],
        )

        llm_helper = LLMHelper()
        keyword_search = kwargs.get("keywords", [])

        results = self.vector_store.client.search(
            search_text=keyword_search,
            search_fields=["keywords"],
            vector_queries=[
                VectorizedQuery(
                    vector=np.array(self.vector_store.embed_query(
                        question), dtype=np.float32).tolist(),
                    k_nearest_neighbors=self.search_k_nearest_neighbors,
                    fields=FIELDS_CONTENT_VECTOR,
                )
            ],
        )
        
        filtered_results = []
        for result in results:
            score = result.get('@search.score', 0)
            is_skipped = score < self.search_score_threshold
            print((f"ID: {result['id']}, Source: {result['source']} Score: {score} Skipped: {is_skipped}"))
            if not is_skipped:
                filtered_results.append(result)
        
        # Retrieve documents as sources
        sources = [Document(
            page_content=result.pop(FIELDS_CONTENT),
            metadata=json.loads(result[FIELDS_METADATA])
            if FIELDS_METADATA in result
            else {
                k: v for k, v in result.items() if k != FIELDS_CONTENT_VECTOR
            },
        ) for result in filtered_results]

        # Generate answer from sources
        answer_generator = LLMChain(
            llm=llm_helper.get_llm(), prompt=answering_prompt, verbose=self.verbose
        )
        sources_text = "\n\n".join(
            [f"[doc{i+1}]: {source.page_content}" for i,
                source in enumerate(sources)]
        )

        with get_openai_callback() as cb:
            result = answer_generator(
                {"question": question, "sources": sources_text})

        answer = result["text"]
        print(f"Answer: {answer}")

        # Generate Answer Object
        source_documents = []
        for source in sources:
            source_document = SourceDocument(
                id=source.metadata["id"],
                content=source.page_content,
                title=source.metadata["title"],
                source=source.metadata["source"],
                chunk=source.metadata["chunk"],
                offset=source.metadata["offset"],
                keywords=source.metadata["keywords"],
                page_number=source.metadata["page_number"],
            )
            source_documents.append(source_document)

        clean_answer = Answer(
            question=question,
            answer=answer,
            source_documents=source_documents,
            prompt_tokens=cb.prompt_tokens,
            completion_tokens=cb.completion_tokens,
        )
        return clean_answer
