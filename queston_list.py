# -*- coding: utf-8 -*-
from pydantic import BaseModel, Field
from typing import List


class QuestionList(BaseModel):
    multimodal_support: bool = Field(
        description="Does the method support multi-modal documents that include text, images, or tables?",
    )
    modalities: str = Field(
        description="What modalities are addressed? (e.g., text, image, table, etc.)",
    )
    llm: str = Field(
        description="What type(s) of LLM(s) were used to validate the method?",
    )
    data: str = Field(
        description="What type(s) of data were used to validate the method?",
    )
    integration: str = Field(
        description="How does the method handle multiple modalities?",
    )
    characteristic: List[str] = Field(
        description="What were the top three characteristics of the study?",
    )
    retrieval_challenge: str = Field(
        description=(
            "Did the paper provide insights or solutions for addressing challenges "
            "in retrieving specific information from large-scale, multi-modal documents (including text, images, and/or tables)? "
            "If so, how?"
        ),
    )
