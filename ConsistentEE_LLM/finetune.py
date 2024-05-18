import os
import sys
from typing import List

# import wandb
import fire
import torch
import transformers
from datasets import load_dataset,load_from_disk
from llama import trainer as T

"""
Unused imports:
import torch.nn as nn
import bitsandbytes as bnb
"""

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)
# from transformers import LlamaForCausalLM, LlamaTokenizer
from llama.modeling_llama import LlamaForCausalLM
from llama.tokenization_llama import LlamaTokenizer



from utils.prompter import Prompter


def train(
    # model/data params
    base_model: str = '/home/liangyunzhen/yhhong/transformers/alpaca-lora-7b-hf_withheads_mode23',
    data_path: str = "/home/zlchen/codes/hyh/alpaca_data_gpt4.json",
    output_dir: str = "/home/liangyunzhen/yhhong/alpaca-lora-ConsistentEE_x/mode23_classify", #
    # training hyperparams
    batch_size: int = 128,
    micro_batch_size: int = 4,
    num_epochs: int = 1,
    learning_rate: float = 3e-4,
    cutoff_len: int = 256,
    val_set_size: int = 2000,
    # lora hyperparams
    lora_r: int = 16,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: List[str] = [
        # "q_proj",
        # "v_proj",
        "lm_heads_0",
        "lm_heads_1",
        "lm_heads_2",
        "lm_heads_3",
        "lm_heads_4",
        "lm_heads_5",
        "lm_heads_6",
        "lm_heads_7",
        "lm_heads_8",
        "lm_heads_9",
        "lm_heads_10",
        "lm_heads_11",
        "lm_heads_12",
        "lm_heads_13",
        "lm_heads_14",
        "lm_heads_15",
        "lm_heads_16",
        "lm_heads_17",
        "lm_heads_18",
        "lm_heads_19",
        "lm_heads_20",
        "lm_heads_21",
        "lm_heads_22",
        "lm_heads_23",
        "lm_heads_24",
        "lm_heads_25",
        "lm_heads_26",
        "lm_heads_27",
        "lm_heads_28",
        "lm_heads_29",
        "lm_heads_30",
        "lm_heads_31",
        # policy
        # "k_proj",
        # "o_proj",
    ],
    # llm hyperparams
    train_on_inputs: bool = True,  # if False, masks out inputs in loss
    add_eos_token: bool = True,
    group_by_length: bool = False,  # faster, but produces an odd training loss curve
    # wandb params
    wandb_project: str = "",
    wandb_run_name: str = "",
    wandb_watch: str = "",  # options: false | gradients | all
    wandb_log_model: str = "",  # options: false | true
    resume_from_checkpoint: str = None,#"/home/liangyunzhen/yhhong/alpaca-lora-main/tolen_alpaca_7b",
    prompt_template_name: str = "alpaca",  # The prompt template to use, will default to alpaca.
    mode: int = 1, # The training mode.
    dataset_name: str = "alpaca",
    alpha:float = 0.00001
):
    if mode == 1:
        resume_from_checkpoint = False
    elif mode == 2:
        lora_target_modules = ["Policy_networks_0", "Policy_networks_1", "Policy_networks_2", "Policy_networks_3",
                               "Policy_networks_4", "Policy_networks_5", "Policy_networks_6", "Policy_networks_7",
                               "Policy_networks_8", "Policy_networks_9",
                               "Policy_networks_10", "Policy_networks_11", "Policy_networks_12", "Policy_networks_13",
                               "Policy_networks_14", "Policy_networks_15", "Policy_networks_16", "Policy_networks_17",
                               "Policy_networks_18", "Policy_networks_19",
                               "Policy_networks_20", "Policy_networks_21", "Policy_networks_22", "Policy_networks_23",
                               "Policy_networks_24", "Policy_networks_25", "Policy_networks_26", "Policy_networks_27",
                               "Policy_networks_28", "Policy_networks_29", "Policy_networks_30", "Policy_networks_31"]
        resume_from_checkpoint = False #'/home/liangyunzhen/yhhong/alpaca-lora-ConsistentEE_x/mode23_0.0/checkpoint-6' #False #'/home/liangyunzhen/yhhong/alpaca-lora-ConsistentEE_x/mode23_0.0/checkpoint-10' #False  # "/home/liangyunzhen/yhhong/alpaca-lora-ConsistentEE/lora4-alpaca_onlyhead_8bitfalse_6epoch"
        #learning_rate = 3e-4
    elif mode == 3:
        lora_target_modules = [
                               "lm_heads_0", "lm_heads_1", "lm_heads_2", "lm_heads_3", "lm_heads_4", "lm_heads_5",
                               "lm_heads_6", "lm_heads_7", "lm_heads_8", "lm_heads_9", "lm_heads_10", "lm_heads_11",
                               "lm_heads_12", "lm_heads_13", "lm_heads_14", "lm_heads_15", "lm_heads_16", "lm_heads_17",
                               "lm_heads_18", "lm_heads_19", "lm_heads_20", "lm_heads_21", "lm_heads_22", "lm_heads_23",
                               "lm_heads_24", "lm_heads_25", "lm_heads_26", "lm_heads_27", "lm_heads_28", "lm_heads_29",
                               "lm_heads_30", "lm_heads_31"]
        resume_from_checkpoint = False
        #learning_rate = 3e-5

    if dataset_name == 'CNN_DM':
        base_model = '/home/liangyunzhen/yhhong/transformers/llama-7b-hf'
        output_dir = "/home/liangyunzhen/yhhong/alpaca-lora-ConsistentEE/lora-cnndm"
        learning_rate = 3e-4
        resume_from_checkpoint = "/home/liangyunzhen/yhhong/alpaca-lora-main/tolen_alpaca_7b"
        num_epochs = 3
        lora_target_modules = ["q_proj","v_proj",
                                 "lm_heads_0",
                                 "lm_heads_1",
                                 "lm_heads_2",
                                 "lm_heads_3",
                                 "lm_heads_4",
                                 "lm_heads_5",
                                 "lm_heads_6",
                                 "lm_heads_7",
                                 "lm_heads_8",
                                 "lm_heads_9",
                                 "lm_heads_10",
                                 "lm_heads_11",
                                 "lm_heads_12",
                                 "lm_heads_13",
                                 "lm_heads_14",
                                 "lm_heads_15",
                                 "lm_heads_16",
                                 "lm_heads_17",
                                 "lm_heads_18",
                                 "lm_heads_19",
                                 "lm_heads_20",
                                 "lm_heads_21",
                                 "lm_heads_22",
                                 "lm_heads_23",
                                 "lm_heads_24",
                                 "lm_heads_25",
                                 "lm_heads_26",
                                 "lm_heads_27",
                                 "lm_heads_28",
                                 "lm_heads_29",
                                 "lm_heads_30",
                                 "lm_heads_31"]



    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(
            f"Training Alpaca-LoRA model with params:\n"
            f"base_model: {base_model}\n"
            f"data_path: {data_path}\n"
            f"output_dir: {output_dir}\n"
            f"batch_size: {batch_size}\n"
            f"micro_batch_size: {micro_batch_size}\n"
            f"num_epochs: {num_epochs}\n"
            f"learning_rate: {learning_rate}\n"
            f"cutoff_len: {cutoff_len}\n"
            f"val_set_size: {val_set_size}\n"
            f"lora_r: {lora_r}\n"
            f"lora_alpha: {lora_alpha}\n"
            f"lora_dropout: {lora_dropout}\n"
            f"lora_target_modules: {lora_target_modules}\n"
            f"train_on_inputs: {train_on_inputs}\n"
            f"add_eos_token: {add_eos_token}\n"
            f"group_by_length: {group_by_length}\n"
            f"wandb_project: {wandb_project}\n"
            f"wandb_run_name: {wandb_run_name}\n"
            f"wandb_watch: {wandb_watch}\n"
            f"wandb_log_model: {wandb_log_model}\n"
            f"resume_from_checkpoint: {resume_from_checkpoint or False}\n"
            f"prompt template: {prompt_template_name}\n"
            f"mode: {mode}\n"
            f"dataset_name: {dataset_name}\n"
            f"alpha: {alpha}\n"
        )
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"


    # wandb usage
    # wandb.init(project="ConsistentEE_run01",
    #             config = {
    #                 "base_model": base_model,
    #                 "data_path": data_path,
    #                 "output_dir": output_dir,
    #                 "batch_size": batch_size,
    #                 "micro_batch_size": micro_batch_size,
    #                 "num_epochs": num_epochs,
    #                 "learning_rate": learning_rate,
    #                 "cutoff_len": cutoff_len,
    #                 "val_set_size": val_set_size,
    #                 "lora_r": lora_r,
    #                 "lora_alpha": lora_alpha,
    #                 "lora_dropout": lora_dropout,
    #                 "lora_target_modules": lora_target_modules,
    #                 "train_on_inputs": train_on_inputs,
    #                 "add_eos_token": add_eos_token,
    #                 "group_by_length": group_by_length,
    #                 "resume_from_checkpoint": resume_from_checkpoint or False,
    #                 "prompt template": prompt_template_name,
    #                 "mode": mode,
    #                "dataset_name": dataset_name,}
    #            )
    # # 备份你的代码
    # wandb.run.log_code('./', include_fn=lambda path: path.endswith(".py") or path.endswith(".ipynb"))


    gradient_accumulation_steps = batch_size // micro_batch_size

    prompter = Prompter(prompt_template_name)


    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    # Check if parameter passed or if set within environ
    # use_wandb = len(wandb_project) > 0 or (
    #     "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
    # )
    # # Only overwrite environ if wandb param passed
    # if len(wandb_project) > 0:
    #     os.environ["WANDB_PROJECT"] = wandb_project
    # if len(wandb_watch) > 0:
    #     os.environ["WANDB_WATCH"] = wandb_watch
    # if len(wandb_log_model) > 0:
    #     os.environ["WANDB_LOG_MODEL"] = wandb_log_model

    model = LlamaForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=False, #True
        torch_dtype=torch.float16,
        device_map=device_map,
    )

    # if mode == 1:
    #     with torch.no_grad():
    #         for lm_head in [getattr(model, f"lm_heads_{i}") for i in range(32)]:
    #             lm_head.weight.copy_(
    #                 model.lm_head.weight)  # 在mode=1的时候需要这个东西    #应该不用这个，感觉还是需要一开始第一阶段的finetune每一层的没分类器，而不能用这个东西


    tokenizer = LlamaTokenizer.from_pretrained(base_model)

    tokenizer.pad_token_id = (
        0  # unk. we want this to be different from the eos token
    )
    tokenizer.padding_side = "left"  # Allow batched inference
    print(f"tokenizer.eos_token_id:{tokenizer.eos_token_id}")

    def tokenize(prompt, add_eos_token=True):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        # print(add_eos_token)
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result

    def generate_and_tokenize_prompt(data_point):
        full_prompt = prompter.generate_prompt(
            data_point["instruction"],
            data_point["input"],
            data_point["output"],
        )
        tokenized_full_prompt = tokenize(full_prompt)
        if not train_on_inputs:
            user_prompt = prompter.generate_prompt(
                data_point["instruction"], data_point["input"]
            )
            tokenized_user_prompt = tokenize(
                user_prompt, add_eos_token=add_eos_token
            )
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            if add_eos_token:
                user_prompt_len -= 1

            tokenized_full_prompt["labels"] = [
                -100
            ] * user_prompt_len + tokenized_full_prompt["labels"][
                user_prompt_len:
            ]  # could be sped up, probably
        return tokenized_full_prompt

    def generate_and_tokenize_prompt_cnndm(data_point):
        full_prompt = prompter.generate_prompt(
            instruction=data_point["article"],
            label=data_point["highlights"],
        )
        tokenized_full_prompt = tokenize(full_prompt)

        return tokenized_full_prompt

    # if mode == 1:
    #     model = prepare_model_for_int8_training(model) #考虑在mode2和3的时候不使用这个，而是直接finetune

    model = prepare_model_for_int8_training(model)  # 考虑在mode1,2和3都使用lora


    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, config)


    if dataset_name == 'alpaca':
        if data_path.endswith(".json") or data_path.endswith(".jsonl"):
            data = load_dataset("json", data_files=data_path)
        else:
            data = load_dataset(data_path)
    elif dataset_name == 'CNN_DM':
        data = load_from_disk('/home/liangyunzhen/yhhong/alpaca-lora-main/dataset/CNN_DM')


    if resume_from_checkpoint:
        # Check the available weights and load them
        checkpoint_name = os.path.join(
            resume_from_checkpoint, "pytorch_model.bin"
        )  # Full checkpoint
        if not os.path.exists(checkpoint_name):
            checkpoint_name = os.path.join(
                resume_from_checkpoint, "adapter_model.bin"
            )  # only LoRA model - LoRA config above has to fit
            resume_from_checkpoint = (
                False  # So the trainer won't try loading its state
            )
        # The two files above have a different name depending on how they were saved, but are actually the same.
        if os.path.exists(checkpoint_name):
            print(f"Restarting from {checkpoint_name}")
            adapters_weights = torch.load(checkpoint_name)
            set_peft_model_state_dict(model, adapters_weights)
        else:
            print(f"Checkpoint {checkpoint_name} not found")

    model.print_trainable_parameters()  # Be more transparent about the % of trainable params.

    if dataset_name == 'alpaca':
        if val_set_size > 0:
            train_val = data["train"].train_test_split(
                test_size=val_set_size, shuffle=True, seed=42
            )
            train_data = (
                train_val["train"].shuffle().map(generate_and_tokenize_prompt)
            )
            val_data = (
                train_val["test"].shuffle().map(generate_and_tokenize_prompt)
            )
        else:
            train_data = data["train"].shuffle().map(generate_and_tokenize_prompt)
            val_data = None
    elif dataset_name == 'CNN_DM':
        train_data = (data["train"].map(generate_and_tokenize_prompt_cnndm))
        val_data = (data["validation"].map(generate_and_tokenize_prompt_cnndm))
        test_data = (data['test'].map(generate_and_tokenize_prompt_cnndm)) #只有在后续的evaluate才用得上



    if not ddp and torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True


    # wandb.watch(model)

    trainer = T.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=100,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            fp16=True,
            logging_steps=10,
            optim="adamw_torch",
            evaluation_strategy="steps" if val_set_size > 0 else "no",
            save_strategy="steps",
            eval_steps=500 if val_set_size > 0 else None, #10
            save_steps=500,
            output_dir=output_dir,
            save_total_limit=3,
            load_best_model_at_end=True if val_set_size > 0 else False,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
            # report_to="wandb" if use_wandb else None,
            # run_name=wandb_run_name if use_wandb else None,
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
        mode=mode,  # 控制在哪个阶段train
        alpha=alpha,#控制退出参数
    )
    model.config.use_cache = False


    # 可能问题是在这里？？
    # if mode == 1:
    #     old_state_dict = model.state_dict
    #     model.state_dict = (
    #         lambda self, *_, **__: get_peft_model_state_dict(
    #             self, old_state_dict()
    #         )
    #     ).__get__(model, type(model))
    # old_state_dict = model.state_dict
    # model.state_dict = (
    #     lambda self, *_, **__: get_peft_model_state_dict(
    #         self, old_state_dict()
    #     )
    # ).__get__(model, type(model))



    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    trainer.train(resume_from_checkpoint=resume_from_checkpoint, mode=mode,alpha=alpha)

    model.save_pretrained(output_dir)
    #print('lora weights saving successfully.')


    """
    print('type(model): ',type(model))

    #假如不使用lora finetune的方法，而是使用直接finetune classifier+policy的方法，然后直接读取这部分的参数
    model = PeftModel.from_pretrained(
        model,
        lora_weights,
        torch_dtype=torch.float16,
    )
    model = model.merge_and_unload()
    model.save_pretrained("/home/liangyunzhen/yhhong/transformers/alpaca-lora-7b-hf")
    tokenizer.save_pretrained("/home/liangyunzhen/yhhong/transformers/alpaca-lora-7b-hf")
    """


    print(
        "\n If there's a warning about missing keys above, please disregard :)"
    )

    # wandb.finish()


if __name__ == "__main__":
    fire.Fire(train)
