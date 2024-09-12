# -*- coding: utf-8 -*-
import arxiv
import datetime
import json
import logging
import os
import tempfile
import time
from typing import Any, Dict, Optional


def download_papers_from_arxiv(max_results: Optional[int] = 10) -> str:
    # PDF の保存先
    temp_dir = tempfile.mkdtemp()
    logging.info(f"Saving directory: {temp_dir}")

    # 検索条件
    today = datetime.datetime.now().strftime(format="%Y%m%d")
    yesterday = (datetime.datetime.now() - datetime.timedelta(days=1)).strftime(format="%Y%m%d")

    query = f'all:"Retrieval Augmented Generation" AND submittedDate:[{yesterday} TO {today}]'
    logging.info(f"Query: {query}")

    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
    )

    # 検索
    client = arxiv.Client()
    results = client.results(search)

    # PDF の保存
    cnt = 0
    for i, paper in enumerate(results):
        logging.info(f"{i}: {paper.title}")

        _id = paper.entry_id.split("/")[-1]  # like "2409.06450v1"
        paper.download_pdf(dirpath=temp_dir, filename=f"{_id}.pdf")

        info = {
            "id": paper.entry_id,
            "title": paper.title,
            "published": paper.published.isoformat(),  # ISOフォーマットで日付を保存
            "summary": paper.summary,
            "categories": paper.categories,
            "journal_ref": paper.journal_ref,
        }
        with open(os.path.join(temp_dir, f"{_id}.json"), "w") as f:
            json.dump(info, f, indent=4)

        cnt += 1
        time.sleep(3)

    logging.info(f"Number of downloaded papers: {cnt}")
    return temp_dir


def get_paper_info(json_path: str) -> Dict[str, Any]:
    with open(json_path, "r") as file:
        info = json.load(file)
    info = {k: v for k, v in info.items()}
    return info


if __name__ == "__main__":
    temp_dir = download_papers_from_arxiv(max_results=1)
