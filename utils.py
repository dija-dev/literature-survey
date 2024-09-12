# -*- coding: utf-8 -*-
import logging
import torch
from langchain.llms import OpenAI, HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
from typing import Optional


def setup_llm(
    hf_model_id: str = "elyza/Llama-3-ELYZA-JP-8B", quantization: Optional[bool] = True, max_tokens: Optional[int] = 500
):
    """
    動作確認済みの hf_model_id:
        elyza/Llama-3-ELYZA-JP-8B
        matsuo-lab/weblab-10b-instruction-sft
        meta-llama/Llama-2-13b-chat-hf
    """

    # hf_model_id が None の場合は OpenAI の LLM を返す
    if hf_model_id is None:
        return OpenAI(temperature=0, max_tokens=max_tokens)

    tokenizer = AutoTokenizer.from_pretrained(hf_model_id)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,  # 量子化の有効化
        bnb_4bit_quant_type="nf4",  # 量子化種別 (fp4 or nf4)
        bnb_4bit_compute_dtype=torch.float16,  # 量子化の dtype (float16 or bfloat16)
        bnb_4bit_use_double_quant=True,  # 二重量子化の有効化
    )

    model = AutoModelForCausalLM.from_pretrained(
        hf_model_id,
        torch_dtype=torch.float16,
        quantization_config=bnb_config if quantization else None,
        device_map="auto",
    )

    hf_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_tokens,
        device_map="auto",
    )

    return HuggingFacePipeline(pipeline=hf_pipeline)


if __name__ == "__main__":
    llm = setup_llm()
    query = "月見を題材に詩を詠んでください。"

    output = llm.invoke(query)
    logging.info(output)
