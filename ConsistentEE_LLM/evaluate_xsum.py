#dolly
import os
import json

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
from llama.modeling_llama import get_global_tokens_layers
from utils.callbacks import Iteratorize, Stream
from utils.prompter import Prompter
import numpy as np



if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:  # noqa: E722
    pass


# Bleu Score Calculation
def calculate_bleu(reference, candidate):
    reference = [reference.split()]
    candidate = candidate.split()
    bleu_score = sentence_bleu(reference, candidate)
    return bleu_score

# Rouge-L Score Calculation
def calculate_rouge_l(reference, candidate):
    rouge = Rouge()
    scores = rouge.get_scores(candidate, reference)
    rouge_l_score = scores[0]['rouge-l']['f']
    return rouge_l_score

# Bert-F Score Calculation
def calculate_bert_f(reference, candidate):
    _, _, bert_scores = score([candidate], [reference], lang="en", model_type="bert-base-uncased")
    bert_f_score = bert_scores[0]  # Extracting the F1 score
    return bert_f_score



def main(
    load_8bit: bool = False, #decapoda-research/llama-7b-hf
    base_model: str = 'Llama2-7b-hf', 
    head_dir: str = None ,
    lora_weights: str = None,
    new_lora_weights: str = None,
    prompt_template: str = "",  
    server_name: str = "0.0.0.0",  
    share_gradio: bool = False,
    output_dir: str = '/home/zlchen/codes/consistentEE/alpaca-lora-ConsistentEE/result'
):
    base_model = base_model or os.environ.get("BASE_MODEL", "")
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"

    prompter = Prompter("CNN_DM")
    tokenizer = LlamaTokenizer.from_pretrained(base_model)

    

   

    if device == "cuda":

        print('using cuda as devices')
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=load_8bit, 
            torch_dtype=torch.float16,
            device_map="auto",
            
        )


        print('type(model): ',type(model))




    # unwind broken decapoda-research config
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    if not load_8bit:
        model.half()  # seems to fix bugs for some users.

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    def evaluate(
        instruction,
        input=None,
        temperature=1,#0.1,
        top_p=0.75,#0.75,
        top_k=40,#40,
        num_beams=1,#4,
        max_new_tokens=128,
        stream_output=False,
        **kwargs,
    ):
        prompt = prompter.generate_prompt(instruction)
        # print('prompt: ',prompt)
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        generation_config = GenerationConfig(
            repetition_penalty=1.2,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            **kwargs,
        )

        generate_params = {
            "input_ids": input_ids,
            "generation_config": generation_config,
            "return_dict_in_generate": True,
            "output_scores": True,
            "max_new_tokens": max_new_tokens,
        }

        if stream_output:
            # Stream the reply 1 token at a time.
            # This is based on the trick of using 'stopping_criteria' to create an iterator,
            # from https://github.com/oobabooga/text-generation-webui/blob/ad37f396fc8bcbab90e11ecf17c56c97bfbd4a9c/modules/text_generation.py#L216-L243.

            def generate_with_callback(callback=None, **kwargs):
                kwargs.setdefault(
                    "stopping_criteria", transformers.StoppingCriteriaList()
                )
                kwargs["stopping_criteria"].append(
                    Stream(callback_func=callback)
                )
                with torch.no_grad():
                    model.generate(**kwargs)

            def generate_with_streaming(**kwargs):
                return Iteratorize(
                    generate_with_callback, kwargs, callback=None
                )

            with generate_with_streaming(**generate_params) as generator:
                for output in generator:
                    # new_tokens = len(output) - len(input_ids[0])
                    decoded_output = tokenizer.decode(output)

                    if output[-1] in [tokenizer.eos_token_id]:
                        break

                    yield prompter.get_response(decoded_output)
            return  # early return for stream_output

        # Without streaming
        with torch.no_grad():
            
            
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )
            
            
        # print("zzzzzzzzzzzzzzzzzz",generation_output)
        s = generation_output.sequences[0]
        output = tokenizer.decode(s)
        #print("output: ",output)
        yield prompter.get_response(output)

    # 读取 JSON 文件
    json_file_path = './data/xsum_0shot.jsonl'  # 替换为您的文件路径
    data = []
    with open(json_file_path, 'r') as f:
        for line in f:
            if line.strip() != '':
                data.append(json.loads(line))
    print("len: ",len(data))
    print(data[1])
    output_data = []
    data = data[:100]

    rouge = Rouge()
    sample_layer=[]
    rouge1_score_list = []
    rouge2_score_list = []
    rougel_score_list = []
    
    for idx, item in tqdm(enumerate(data), total=len(data), desc="Processing"):
        for response in evaluate(instruction = item['article']): #模拟真实情况，一条一条做evaluate
            
            # print("item['instruction']: ",item['instruction'])
            # print('item["input"]: ',item["input"])
            # print("response: ",response)
            tokens_num, layers_num = get_global_tokens_layers(mode=5)#推理模式
            avg_layers=layers_num/tokens_num
            # print(f"test_global:{avg_layers}")
            sample_layer.append(avg_layers)

            output_data.append({"instruction": item['article'],"answer": response})
            scores = rouge.get_scores(response, item['summary_gt'])[0]
            rouge1_score_list.append(scores['rouge-1']['f'])
            rouge2_score_list.append(scores['rouge-2']['f'])
            rougel_score_list.append(scores['rouge-l']['f'])
            print('rouge-1: {:.6f}, rouge-2: {:.6f}, rouge-l: {:.6f},avg_layer: {:.3f}'.format(np.mean(rouge1_score_list), np.mean(rouge2_score_list), np.mean(rougel_score_list), np.mean(sample_layer)))
            #item['alpaca_answer'] = response

            # bleu_score = calculate_bleu(item['output'], response)
            # rouge_l_score = calculate_rouge_l(item['output'], response)
            # bert_f_score = calculate_bert_f(item['output'], response)
            #
            # print("Bleu Score:", bleu_score)
            # print("Rouge-L Score:", rouge_l_score)
            # print("BERT-F Score:", bert_f_score)

    json_data = json.dumps(output_data, indent=2)

    # 指定文件名
    file_name = output_dir

    # 将 JSON 数据写入文件
    with open(file_name, 'w') as file:
        # 分别写入四个值，并在每个值前面加上对应的名字
        file.write("rouge-1: {:.6f}, rouge-2: {:.6f}, rouge-l: {:.6f}, avg_layer: {:.3f}\n".format(
            np.mean(rouge1_score_list),
            np.mean(rouge2_score_list),
            np.mean(rougel_score_list),
            np.mean(sample_layer)
        ))


        file.write(json_data)


            #data[idx]['alpaca_answer'] = response

    # # 写入更新后的数据到新的 JSON 文件
    # new_json_file_path = '/root/autodl-tmp/alpaca-lora-main/dataset/dolly_alpaca_answer.json'
    # with open(new_json_file_path, 'w', encoding='utf-8') as new_json_file:
    #     json.dump(data, new_json_file, indent=4)
    #
    # print("JSON file updated and saved as:", new_json_file_path)

    # # testing code for readme
    # for instruction in [
    #     "Tell me about alpacas.",
    #     "Tell me about the president of Mexico in 2019.",
    #     "Tell me about the king of France in 2019.",
    #     "List all Canadian provinces in alphabetical order.",
    #     "Write a Python program that prints the first 10 Fibonacci numbers.",
    #     "Write a program that prints the numbers from 1 to 100. But for multiples of three print 'Fizz' instead of the number and for the multiples of five print 'Buzz'. For numbers which are multiples of both three and five print 'FizzBuzz'.",  # noqa: E501
    #     "Tell me five words that rhyme with 'shock'.",
    #     "Translate the sentence 'I have no mouth but I must scream' into Spanish.",
    #     "Count up from 1 to 500.",
    # ]:
    #     print("Instruction:", instruction)
    #     for response in evaluate(instruction):
    #         print("response: ",response)
    #     #print("Response:", evaluate(instruction))





#Mobile phone usage negatively affects human socialization



if __name__ == "__main__":
    fire.Fire(main)






