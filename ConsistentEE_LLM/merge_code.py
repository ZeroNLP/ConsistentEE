
import json
import os
import sys
import nltk
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge
from bert_score import score

import fire
import torch
import transformers
from peft import PeftModel
from transformers import GenerationConfig

from llama.modeling_llama import LlamaForCausalLM
from llama.tokenization_llama import LlamaTokenizer

from utils.callbacks import Iteratorize, Stream
from utils.prompter import Prompter

def merge_code(base_model, lora_weights, output_dir,mode):
    # print(base_model,'\t',lora_weights,'\t',output_dir)
    # print(type(base_model))
    model = LlamaForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=False,  # 不使用8bit，会出错
        torch_dtype=torch.float16,
        device_map="auto",
    )
    tokenizer = LlamaTokenizer.from_pretrained(base_model)

    print('type(model): ', type(model))
    if mode == 0:
        print('*********doing*********')
        with torch.no_grad():
            for lm_head in [getattr(model, f"lm_heads_{i}") for i in range(32)]:
                lm_head.weight.copy_(
                    model.lm_head.weight)  # 在mode=1的时候需要这个东西    

    # 假如不使用lora finetune的方法，而是使用直接finetune classifier+policy的方法，然后直接读取这部分的参数
    model = PeftModel.from_pretrained(
        model,
        lora_weights,
        torch_dtype=torch.float16,
    )
    model = model.merge_and_unload()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    fire.Fire(merge_code)
