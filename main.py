# -*- coding: utf-8 -*-
import json
import logging
import os
from langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.output_parsers import CommaSeparatedListOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

# 自作モジュール
from queston_list import QuestionList
from utils import setup_llm
from estimate_applicability import estimate_applicability_from_title
from download_papers import download_papers_from_arxiv, get_paper_info


def extract_paper_details(pdf_path: str, llm, embeddings):
    # 基本情報の読取
    json_path = pdf_path.replace(".pdf", ".json")
    info = get_paper_info(json_path)

    # 論文タイトルからラフに評価
    rough_appl = estimate_applicability_from_title(llm, info["title"])  # like {"answer": "...", "score": "100"}
    info.update(rough_appl)

    # スコアがしきい値以上の場合
    if int(rough_appl["score"]) >= 20:

        qa_chain = load_qa_chain(
            llm=llm,
            chain_type="refine",  # "stuff", "map_reduce", "refine", "map_rerank"
            return_intermediate_steps=True,
            verbose=True,
        )

        output_parser = CommaSeparatedListOutputParser()

        # pdf の読取
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()

        # chunk 分割
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=254)
        chunks = text_splitter.split_documents(pages)

        # db 化
        temp_db = Chroma.from_documents(chunks, embedding=embeddings)

        # retriever 化
        retriever = temp_db.as_retriever(search_type="similarity", search_kwargs={"k": 4})

        result = dict()
        for field in QuestionList.model_fields:
            question = QuestionList.model_fields[field].description  # 質問
            logging.info(f"{question = }")

            contexts = retriever.get_relevant_documents(query=question)  # 検索結果

            # qa chain
            answer = qa_chain({"input_documents": contexts, "question": question}, return_only_outputs=True)
            answer = answer["output_text"]
            if field in ["characteristic"]:
                answer = output_parser.parse(answer)

            result[field] = dict(question=question, answer=answer)

        info.update({"qa": result})

    # 結果の保存
    with open(json_path, "w") as f:
        json.dump(info, f, indent=4)

    return None


if __name__ == "__main__":
    temp_dir = download_papers_from_arxiv(max_results=10)

    llm = setup_llm()
    embeddings = HuggingFaceEmbeddings()

    for fname in os.listdir(temp_dir):
        pdf_path = os.path.join(temp_dir, fname)
        extract_paper_details(pdf_path, llm, embeddings)
