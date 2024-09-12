# -*- coding: utf-8 -*-
import logging
from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from langchain.output_parsers import RegexParser
from typing import Dict

# 自作モジュール
from utils import setup_llm


def estimate_applicability_from_title(llm, title: str) -> Dict[str, str]:
    # create the list of few shot examples
    examples = [
        {
            "title": "Multimodal Retrieval-Augmented Generation for Healthcare",
            "answer": "Yes. This paper explores the use of RAG techniques with both text and medical imaging data, aligning well with multimodal RAG.",
            "score": "100",
        },
        {
            "title": "Enhancing Document Retrieval with Text-Based Models",
            "answer": "No. This paper focuses on text-based retrieval models without incorporating multiple modalities.",
            "score": "20",
        },
        {
            "title": "Combining Text and Visual Data for Improved Question Answering",
            "answer": "Yes. The paper addresses the integration of text and visual data for question answering, relevant to multimodal RAG approaches.",
            "score": "80",
        },
        {
            "title": "Analyzing Speech and Text Data for Sentiment Analysis",
            "answer": "Not clear. While it involves text and speech data, the focus is on sentiment analysis rather than RAG.",
            "score": "10",
        },
        {
            "title": "Developing Multimodal Systems for Autonomous Vehicles",
            "answer": "Yes. This paper involves the integration of various data types (e.g., video, sensor data) for autonomous vehicles, which is relevant to multimodal RAG.",
            "score": "85",
        },
    ]

    # specify the template to format the examples
    example_formatter_template = "Paper Title: {title}\n" "Answer: {answer}\n" "Score: {score}"

    example_prompt = PromptTemplate(
        template=example_formatter_template,
        input_variables=["title", "answer", "score"],
        validate_template=False,
    )

    prefix = (
        "Based on the following title of the paper, please estimate its applicability to multimodal retrieval-augmented generation (RAG).\n"
        "If you do not know, answer that you do not know.\n\n"
        "Paper Title: [title of paper]\n"
        "Answer: [applicability to multimodal RAG]\n"
        "Score: [estimated applicability from 0 to 100]\n\n"
        "How to determine the score\n"
        "- If the paper is presumed to have high potential for application in multimodal RAG, a high score will be given.\n"
        "- Be careful not to be overconfident!\n"
    )
    suffix = "Paper Title: {title}\n" "Answer: "

    few_shot_prompt = FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
        prefix=prefix,  # some text that goes before the examples in the prompt
        suffix=suffix,  # some text that goes after the examples in the prompt
        input_variables=["title"],
        example_separator="\n\n",
    )

    prompt_text = few_shot_prompt.format(title=title.encode("utf-8").decode("utf-8"))
    logging.info(f"{prompt_text = }")

    output = llm.invoke(prompt_text.encode("utf-8").decode("utf-8"))
    logging.info(f"{output = }")

    output_parser = RegexParser(
        regex=r"\s(.*?)\nScore: (.*)",
        output_keys=["answer", "score"],
    )
    format_output = output_parser.parse(output.replace(prompt_text, ""))

    return format_output


if __name__ == "__main__":
    llm = setup_llm()
    titile = "Multimodal Large Language Model Driven Scenario Testing for Autonomous Vehicles"

    format_output = estimate_applicability_from_title(llm, titile)
    logging.info(f"{format_output = }")
