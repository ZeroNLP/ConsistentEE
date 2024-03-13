# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for sequence classification on GLUE (Bert, XLM, XLNet, RoBERTa, Albert, XLM-RoBERTa)."""


import argparse
import glob
import json
import logging
import os
import random
import time
import csv
import itertools
import math

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from transformers import (
    WEIGHTS_NAME,
    AdamW,
    AlbertConfig,
    AlbertTokenizer,
    BertConfig,
    BertTokenizer,
    DistilBertConfig,
    DistilBertForSequenceClassification,
    DistilBertTokenizer,
    FlaubertConfig,
    FlaubertForSequenceClassification,
    FlaubertTokenizer,
    RobertaConfig,
    RobertaForSequenceClassification,
    RobertaTokenizer,
    XLMConfig,
    XLMForSequenceClassification,
    XLMRobertaConfig,
    XLMRobertaForSequenceClassification,
    XLMRobertaTokenizer,
    XLMTokenizer,
    XLNetConfig,
    XLNetForSequenceClassification,
    XLNetTokenizer,
    get_linear_schedule_with_warmup,
)

from consistentee.modeling_bert import BertForSequenceClassification
#from transformers import glue_compute_metrics as compute_metrics
#from transformers import glue_convert_examples_to_features as convert_examples_to_features
from glue_mode2.glue_mode2 import glue_output_modes as output_modes
#from transformers import glue_processors as processors
from glue_mode2.glue_mode2 import glue_processors as processors
from glue_mode2.glue_mode2 import glue_convert_examples_to_features as convert_examples_to_features


try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter


logger = logging.getLogger(__name__)

ALL_MODELS = sum(
    (
        tuple(conf.pretrained_config_archive_map.keys())
        for conf in (
            BertConfig,
            XLNetConfig,
            XLMConfig,
            RobertaConfig,
            DistilBertConfig,
            AlbertConfig,
            XLMRobertaConfig,
            FlaubertConfig,
        )
    ),
    (),
)

MODEL_CLASSES = {
    "bert": (BertConfig, BertForSequenceClassification, BertTokenizer),
    "xlnet": (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
    "xlm": (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
    "roberta": (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
    "distilbert": (DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizer),
    "xlmroberta": (XLMRobertaConfig, XLMRobertaForSequenceClassification, XLMRobertaTokenizer),
    "flaubert": (FlaubertConfig, FlaubertForSequenceClassification, FlaubertTokenizer),
}


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def train(args, train_dataset, model, tokenizer):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
        os.path.join(args.model_name_or_path, "scheduler.pt")
    ):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True,
        )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    if os.path.exists(args.model_name_or_path):
        # set global_step to gobal_step of last saved checkpoint from model path
        global_step = int(args.model_name_or_path.split("-")[-1].split("/")[0])
        epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
        steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)

        logger.info("  Continuing training from checkpoint, will skip to saved global_step")
        logger.info("  Continuing training from epoch %d", epochs_trained)
        logger.info("  Continuing training from global step %d", global_step)
        logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)

    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0],
    )
    set_seed(args)  # Added here for reproductibility
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):

            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3],"mode": args.mode}
            if args.model_type != "distilbert":
                inputs["token_type_ids"] = (
                    batch[2] if args.model_type in ["bert", "xlnet", "albert"] else None
                )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids
            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    logs = {}
                    if (
                        args.local_rank == -1 and args.evaluate_during_training
                    ):  # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate(args, model, tokenizer)
                        for key, value in results.items():
                            eval_key = "eval_{}".format(key)
                            logs[eval_key] = value
                        #evaluate_train_loss(args, model, tokenizer)

                    loss_scalar = (tr_loss - logging_loss) / args.logging_steps
                    learning_rate_scalar = scheduler.get_lr()[0]
                    logs["learning_rate"] = learning_rate_scalar
                    logs["loss"] = loss_scalar
                    logging_loss = tr_loss

                    for key, value in logs.items():
                        tb_writer.add_scalar(key, value, global_step)
                    print(json.dumps({**logs, **{"step": global_step}}))

                # if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                #     # Save model checkpoint
                #     output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                #     if not os.path.exists(output_dir):
                #         os.makedirs(output_dir)
                #     model_to_save = (
                #         model.module if hasattr(model, "module") else model
                #     )  # Take care of distributed/parallel training
                #     model_to_save.save_pretrained(output_dir)
                #     tokenizer.save_pretrained(output_dir)
                #
                #     torch.save(args, os.path.join(output_dir, "training_args.bin"))
                #     logger.info("Saving model checkpoint to %s", output_dir)
                #
                #     torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                #     torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                #     logger.info("Saving optimizer and scheduler states to %s", output_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step

def train_rl(args, train_dataset, model, tokenizer):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)

    train_params = []
    for n, p in model.named_parameters():
        if n.find('Policy_network') != -1:
            train_params.append((n, p))
            if p.requires_grad:
                print('parameters: ', n)
                print('p: ', p)
            else:
                print('no_grad_parameters: ', n)
        else:
            p.requires_grad = False

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in train_params if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in train_params if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    """
    if args.warmup_steps == -1:
        args.warmup_steps = int(t_total * 0.1)
    """
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )



    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True,
        )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0

    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0],
    )
    set_seed(args)  # Added here for reproductibility

    sample_K = 8 ##reuse

    #frequency = [0.04693140794223827, 0.01444043321299639, 0.1371841155234657, 0.3140794223826715, 0.05054151624548736, 0.02527075812274368, 0.01444043321299639, 0.0036101083032490976, 0.018050541516245487, 0.01444043321299639, 0.007220216606498195, 0.35379061371841153]

    #rte #frequency = [0.019897497738920713, 0.0027132951462164605, 0.07898703647874586, 0.137473620741634, 0.17425384383479048, 0.03255954175459753, 0.07958999095568285, 0.023816701839011155, 0.058788061501356646, 0.024419656315948148, 0.02622851974675912, 0.34127223394633704]
    frequency = [0.5935114503816794, 0.0, 0.0, 0.0, 0.0, 0.13004362050163576, 0.004089422028353326, 0.0027262813522355507, 0.04743729552889858, 0.02317339149400218, 0.010359869138495093, 0.18865866957470012]

    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            bsz = batch[0].size(0)

            if sample_K > 1:
                new_batch = []
                for t in batch:
                    tmp = []
                    for j in range(sample_K):
                        tmp.append(t)
                    tmp = torch.cat(tmp, dim=0)
                    new_batch.append(tmp)
                batch = new_batch

            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3],"mode": args.mode}

            if args.model_type != "distilbert":
                inputs["token_type_ids"] = (
                    batch[2] if args.model_type in ["bert", "xlnet", "albert"] else None
                )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids
            outputs = model(**inputs)

            #print(outputs)

            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)
            logits = outputs[1]
            layers = outputs[2]
            probs = outputs[3]

            L = len(loss)
            classfiy_loss = sum(loss) / L

            bsz = L // sample_K
            rewards = [[] for _ in range(bsz)]

            for i in range(L):
                #print('batch[4][i]: ',batch[4][i])
                #print('loss[i]: ',loss[i])
                """
                if batch[5][i]>=9:
                    reward = -loss[i].item() - layers[i] * 0.04* (1-batch[5][i]/12)#(1-batch[4][i])
                else:
                    reward = -loss[i].item() - layers[i] * 0.08*(1-batch[5][i] / 12)
                """
                #reward =  -loss[i].item() - layers[i] * 0.02 * (1 - exit_layer / 12) +  frequency[int(layers[i])-1] * 0.25 # 0.06 0.25
                reward = -loss[i].item() #- layers[i] * 0.015#- layers[i] * 0.015#- layers[i] * 0.09 * (1 - batch[5][i] / 12) #+ frequency[int(layers[i])-1] * 0.1#0.015#0.03 # +  frequency[int(layers[i])-1] * 0.25 # 0.06 0.25
                #reward = -loss[i].item() - layers[i] * 0.03 * (1 - batch[5][i] / 12) + frequency[int(layers[i]) - 1] * 0.25

                #reward = -loss[i].item() - torch.log(layers[i]) * 0.1 * (1 - batch[5][i] / 12)

                #reward = -loss[i].item() - layers[i] * 0.060 * (1 - batch[5][i] / 18)
                #reward = -loss[i].item() - layers[i] * 0.02785#0.024#0.0052#75#args.alpha #layers[i]指的是第i个样本在layers[i]层退出

                """
                if loss[i] < torch.log(torch.tensor(0.5)):  # 说明错误了
                    reward = -loss[i].item() - layers[i] * 0.01 * (1 - batch[5][i] / 12)
                     #- layers[i] * 0.033#-loss[i] + 0.6931  # - layers[i] * 0.2#0.1* (1-batch[4][i])  #RTE: 0.035   #* 0.02785 #- 100*batch[4][i]  # 0.024#0.0052#75#args.alpha #layers[i]指的是第i个样本在layers[i]层退出
                else:
                    reward = -loss[i].item() - layers[i] * 0.01 * (1 - batch[5][i] / 12)#-layers[i] * 0.57 + 0.6931  # 0.6931/12 = 0.05776
                """
                rewards[i % bsz].append(reward)  ##记录下来L个reward，L是一个总batch内所有样本的个数

            #print('loss: ',loss)
            #print('rewards: ',rewards)
            policy_loss = 0
            for i in range(bsz):
                rs = rewards[i]
                #print('rs: ',rs)
                baseline = sum(rs) / len(rs)
                #print('baseline: ',baseline)
                for j in range(sample_K):
                    reward = (rs[j] - baseline)  ##reward是最终的delayed reward
                    policy_loss += reward * probs[j * bsz + i] * -1 ##把每一次的概率都累乘起来？
            policy_loss = policy_loss / L
            loss = policy_loss
            #print('loss: ',loss)


            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            """
            for n, p in model.named_parameters():
                if n.find('Policy_network1') != -1:
                    print(n)
                    print(p)
            """

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    logs = {}
                    if (
                        args.local_rank == -1 and args.evaluate_during_training
                    ):  # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate(args, model, tokenizer)
                        for key, value in results.items():
                            eval_key = "eval_{}".format(key)
                            logs[eval_key] = value

                    loss_scalar = (tr_loss - logging_loss) / args.logging_steps
                    learning_rate_scalar = scheduler.get_lr()[0]
                    logs["learning_rate"] = learning_rate_scalar
                    logs["loss"] = loss_scalar
                    logging_loss = tr_loss

                    for key, value in logs.items():
                        tb_writer.add_scalar(key, value, global_step)
                    print(json.dumps({**logs, **{"step": global_step}}))

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = (
                        model.module if hasattr(model, "module") else model
                    )  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)

                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    logger.info("Saving model checkpoint to %s", output_dir)

                    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                    logger.info("Saving optimizer and scheduler states to %s", output_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step

def train_both(args, train_dataset, model, tokenizer):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)


    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)

    if args.warmup_steps == -1:
        args.warmup_steps = int(t_total * 0.1)

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )


    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True,
        )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0

    tr_loss, logging_loss = 0.0, 0.0
    tr_policy_loss, logging_policy_loss = 0.0, 0.0

    model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0],
    )
    set_seed(args)  # Added here for reproductibility

    sample_K = 8  ##reuse

    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            bsz = batch[0].size(0)

            if sample_K > 1:
                new_batch = []
                for t in batch:
                    tmp = []
                    for j in range(sample_K):
                        tmp.append(t)
                    tmp = torch.cat(tmp, dim=0)
                    new_batch.append(tmp)
                batch = new_batch

            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3], "mode": args.mode}

            if args.model_type != "distilbert":
                inputs["token_type_ids"] = (
                    batch[2] if args.model_type in ["bert", "xlnet", "albert"] else None
                )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids
            outputs = model(**inputs)

            # print(outputs)

            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)
            layers = outputs[2]
            probs = outputs[3]

            L = len(loss)
            classfiy_loss = sum(loss) / L

            bsz = L // sample_K
            rewards = []
            for i in range(bsz):
                rewards.append([])

            for i in range(L):                                             #batch[4][i]是指对应样本的原loss值，这个因素，是为了将样本与样本之间更加区分开来

                reward = -loss[i].item() #- layers[i] * 0.02 * (1 - batch[5][i] / 12)#- layers[i] * 0.015-  #* 0.025 # args.alpha #layers[i]指的是第i个样本在layers[i]层退出
                rewards[i % bsz].append(reward)  ##记录下来L个reward，L是一个总batch内所有样本的个数

            policy_loss = 0
            for i in range(bsz):
                rs = rewards[i]
                baseline = sum(rs) / len(rs)
                # print('baseline: ',baseline)
                for j in range(sample_K):
                    reward = (rs[j] - baseline)  ##reward是最终的delayed reward
                    policy_loss += reward * probs[j * bsz + i] * -1  ##把每一次的概率都累乘起来？
            policy_loss = policy_loss / L
            loss = classfiy_loss + policy_loss
            #print('loss: ', loss)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += classfiy_loss.item()
            tr_policy_loss += policy_loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    logs = {}
                    if (
                            args.local_rank == -1 and args.evaluate_during_training
                    ):  # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate(args, model, tokenizer)
                        for key, value in results.items():
                            eval_key = "eval_{}".format(key)
                            logs[eval_key] = value

                    loss_scalar = (tr_loss - logging_loss) / args.logging_steps
                    learning_rate_scalar = scheduler.get_lr()[0]
                    logs["learning_rate"] = learning_rate_scalar
                    logs["loss"] = loss_scalar
                    logging_loss = tr_loss

                    policy_loss_scalar = (tr_policy_loss - logging_policy_loss) / args.logging_steps
                    logs["policy_loss"] = policy_loss_scalar
                    logging_policy_loss = tr_policy_loss

                    for key, value in logs.items():
                        tb_writer.add_scalar(key, value, global_step)
                    print(json.dumps({**logs, **{"step": global_step}}))

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = (
                        model.module if hasattr(model, "module") else model
                    )  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)

                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    logger.info("Saving model checkpoint to %s", output_dir)

                    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                    logger.info("Saving optimizer and scheduler states to %s", output_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

        if args.local_rank in [-1, 0]:
            tb_writer.close()

        return global_step, tr_loss / global_step

def train_alternate(args, train_dataset, model, tokenizer):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)

    train_params = []
    for n, p in model.named_parameters():
        if n.find('linear_1') != -1 or n.find('linear_2') != -1 or n.find('linear_3') != -1 \
                or n.find('linear_4') != -1 or n.find('linear_5') != -1 \
                or n.find('linear_6') != -1 or n.find('linear_7') != -1 \
                or n.find('linear_7') != -1 or n.find('linear_8') != -1:
            train_params.append((n, p))
            print(n)
        else:
            p.requires_grad = False

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in train_params if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in train_params if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    """
    if args.warmup_steps == -1:
        args.warmup_steps = int(t_total * 0.1)
    """
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )



    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True,
        )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0



    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0],
    )
    set_seed(args)  # Added here for reproductibility

    sample_K = 8 ##reuse

    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            bsz = batch[0].size(0)

            if sample_K > 1:
                new_batch = []
                for t in batch:
                    tmp = []
                    for j in range(sample_K):
                        tmp.append(t)
                    tmp = torch.cat(tmp, dim=0)
                    new_batch.append(tmp)
                batch = new_batch



            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3],"mode": args.mode}

            if args.model_type != "distilbert":
                inputs["token_type_ids"] = (
                    batch[2] if args.model_type in ["bert", "xlnet", "albert"] else None
                )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids
            outputs = model(**inputs)

            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)
            layers = outputs[2]
            probs = outputs[3]



            L = len(loss)
            classfiy_loss = sum(loss) / L

            bsz = L // sample_K
            rewards = []
            for i in range(bsz):
                rewards.append([])
            dLs = []

            for i in range(L):
                """
                delta_L = 0
                for j in range(reduction_times):
                    delta_L += lens[j][i].item()  ##lens是一个矩阵 L是样本个数 reduction_times 相当于层数   
                delta_L /= reduction_times
                """
                reward = -loss[i].item() - layers[i] * args.alpha #layers[i]指的是第i个样本在layers[i]层退出
                rewards[i % bsz].append(reward)  ##记录下来L个reward，L是一个总batch内所有样本的个数

                """
                dLs.append(delta_L)
            mean_dl = sum(dLs) / len(dLs)
            tr_dL += mean_dl / args.gradient_accumulation_steps
            """

            policy_loss = 0
            for i in range(bsz):
                rs = rewards[i]
                rs = np.array(rs)
                baseline = np.mean(rs)
                for j in range(sample_K):
                    reward = (rs[j] - baseline)  ##reward是最终的delayed reward
                    policy_loss += reward * probs[j * bsz + i] ##把每一次的概率都累乘起来？
            policy_loss = policy_loss / L
            loss = policy_loss






            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()




            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    logs = {}
                    if (
                        args.local_rank == -1 and args.evaluate_during_training
                    ):  # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate(args, model, tokenizer)
                        for key, value in results.items():
                            eval_key = "eval_{}".format(key)
                            logs[eval_key] = value

                    loss_scalar = (tr_loss - logging_loss) / args.logging_steps
                    learning_rate_scalar = scheduler.get_lr()[0]
                    logs["learning_rate"] = learning_rate_scalar
                    logs["loss"] = loss_scalar
                    logging_loss = tr_loss

                    for key, value in logs.items():
                        tb_writer.add_scalar(key, value, global_step)
                    print(json.dumps({**logs, **{"step": global_step}}))

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = (
                        model.module if hasattr(model, "module") else model
                    )  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)

                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    logger.info("Saving model checkpoint to %s", output_dir)

                    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                    logger.info("Saving optimizer and scheduler states to %s", output_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step





def evaluate(args, model, tokenizer, prefix="", patience=0):

    if args.model_type == 'albert':
        model.albert.set_regression_threshold(args.regression_threshold)
        model.albert.set_patience(patience)
        model.albert.reset_stats()
    elif args.model_type == 'bert':
        model.bert.set_regression_threshold(args.regression_threshold)
        model.bert.set_patience(patience)
        model.bert.reset_stats()
    else:
        raise NotImplementedError()

    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_task_names = ("mnli", "mnli-mm") if args.task_name == "mnli" else (args.task_name,)
    eval_outputs_dirs = (args.output_dir, args.output_dir + "-MM") if args.task_name == "mnli" else (args.output_dir,)

    results = {}
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        eval_dataset = load_and_cache_examples(args, eval_task, tokenizer, evaluate=True)

        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

        # multi-gpu eval
        if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
            model = torch.nn.DataParallel(model)

        # Eval!
        logger.info("***** Running evaluation {} *****".format(prefix))
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        loss_respective_all = None
        all_layers = []
        labels_layer = [[] for _ in range(12)]

        start = time.time()
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3],"mode": args.mode}
                if args.model_type != "distilbert":
                    inputs["token_type_ids"] = (
                        batch[2] if args.model_type in ["bert", "xlnet", "albert"] else None
                    )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids

                outputs = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]

                # 只有在mode1才有求loss_respective
                if args.mode == 1 or args.mode == 3:
                    loss_respective = outputs[2]

                    if loss_respective_all is None:
                        loss_respective_all = loss_respective.tolist()
                    else:
                        loss_respective_all += loss_respective.tolist()

                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs["labels"].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

            if args.mode == 3:
                for ix, logits_item in enumerate(outputs[3]):
                    labels_layer[ix].append(np.argmax(logits_item.detach().cpu().numpy(), axis=-1))

            if args.mode == 1:  ##为了给出train_exit才用上这个
                labels_all = None  ###新添加，用来储存所有预测的labels

                for ix, logits_item in enumerate(outputs[3]):
                    if labels_all is None:
                        labels_all = logits_item.argmax(dim=1).unsqueeze(0)
                    else:
                        labels_all = torch.cat((labels_all, logits_item.argmax(dim=1).unsqueeze(0)), dim=0)

                ###关键之步
                # print('labels_all: ',labels_all)
                dim0, dim1 = labels_all.shape  ###没错，这里是基于记忆持续性
                #print('dim0,dim1:',dim0,dim1)   #dim是batchsize
                sample_layers = [12] * dim1  ###从1层开始，十二层为最后一层 普通bert
                for j in range(dim1):
                    for i in range(dim0):
                        label_single = labels_all[dim0 - i - 1][j]
                        if (label_single == inputs["labels"].detach().cpu().numpy()[j]):
                            sample_layers[j] = dim0 - i
                        else:
                            break  ###这里是基于持续性
                print('sample_layers: ', sample_layers)
                """
                #Oracle
                dim0, dim1 = labels_all.shape  ###没错，这里是基于记忆持续性
                sample_layers = [12] * dim1  ###从1层开始，十二层为最后一层 普通bert
                for j in range(dim1):
                    for i in range(dim0):
                        label_single = labels_all[dim0 - i - 1][j]
                        if (label_single == inputs["labels"].detach().cpu().numpy()[j]):
                            sample_layers[j] = dim0 - i
                """

                all_layers.append(sample_layers)
        print("Time taken: ", time.time() - start)

        if args.mode == 3:
            print('evaluate_loss_respective_all: ',loss_respective_all)

        if args.mode == 1 or args.mode == 3:  ##为了给出train_loss才用上这个
            all_layers = list(itertools.chain.from_iterable(all_layers))
            print('all_layers:　', all_layers)
            print('len(loss_respective_all): ', len(loss_respective_all))
            print('loss_respective_all: ',loss_respective_all)

            for id, layer_label in enumerate(labels_layer):
                layer_label = list(itertools.chain.from_iterable(layer_label))
                print('layer_label: ',layer_label )
                print('out_label_ids: ', out_label_ids)
                print(id, "  acc: ", (layer_label == out_label_ids).mean())

            rows_to_write = []
            rows_to_write_exit = []

            if args.task_name != 'mnli' and args.task_name != 'mnli-mm':
                dev_str = "dev.tsv"
            elif args.task_name == 'mnli':
                dev_str = "dev_matched.tsv"
            else:
                dev_str = "dev_matched.tsv"

            with open(os.path.join(args.data_dir, dev_str), 'r', encoding="utf-8-sig") as input_file:
                reader = list(csv.reader(input_file, delimiter="\t", quotechar=None))

                if args.task_name == 'sst-2':
                    for ix, row in enumerate(reader):
                        # 假设要写入第2列和第4列的数据，可以按如下方式选择这些数据：
                        if ix == 0:
                            data_to_write = [row[0], row[1]]  ##写入列名
                            data_to_write_exit = [row[0], row[1]]
                            print('row[1],row[2]: ', data_to_write)

                        else:
                            # print(ix - 1, " ", loss_respective_all[ix - 1])
                            data_to_write = [row[0], loss_respective_all[ix - 1]]  # 对于sst-2任务这种的写入方式  记得ix要减1位，对于
                            data_to_write_exit = [row[0], all_layers[ix - 1]]
                        rows_to_write.append(data_to_write)
                        rows_to_write_exit.append(data_to_write_exit)
                elif args.task_name == 'rte':
                    for ix, row in enumerate(reader):
                        # 假设要写入第2列和第4列的数据，可以按如下方式选择这些数据：
                        if ix == 0:
                            data_to_write = [row[0], row[1], row[2], row[3]]  ##写入列名
                            data_to_write_exit = [row[0], row[1], row[2], row[3]]
                            print('row[1],row[2]: ', data_to_write)
                        else:
                            # print(ix - 1, " ", loss_respective_all[ix - 1])
                            data_to_write = [row[0], row[1], row[2], loss_respective_all[ix - 1]]
                            data_to_write_exit = [row[0], row[1], row[2], all_layers[ix - 1]]
                        rows_to_write.append(data_to_write)
                        rows_to_write_exit.append(data_to_write_exit)
                elif args.task_name == 'mrpc':
                    for ix, row in enumerate(reader):
                        # 假设要写入第2列和第4列的数据，可以按如下方式选择这些数据：
                        if ix == 0:
                            data_to_write = [row[0], row[1], row[2], row[3], row[4]]  ##写入列名
                            data_to_write_exit = [row[0], row[1], row[2], row[3], row[4]]
                            print('row[1],row[2]: ', data_to_write)
                        else:
                            # print(ix,' len(row): ',len(row))
                            # print('row: ',row)
                            # print('len(loss_respective_all): ', len(loss_respective_all))
                            data_to_write = [row[1], row[2], row[3], row[4],
                                             loss_respective_all[ix - 1]]  # , row[2], row[3], row[4]]
                            data_to_write_exit = [row[1], row[2], row[3], row[4], all_layers[ix - 1]]
                        rows_to_write.append(data_to_write)
                        rows_to_write_exit.append(data_to_write_exit)

                elif args.task_name == 'qqp':
                    for ix, row in enumerate(reader):
                        # 假设要写入第2列和第4列的数据，可以按如下方式选择这些数据：
                        if ix == 0:
                            data_to_write = [row[0], row[1], row[2], row[3], row[4], row[5]]  ##写入列名
                            data_to_write_exit = [row[0], row[1], row[2], row[3], row[4], row[5]]
                            print('row[0], row[1], row[2], row[3], row[4], row[5]: ', data_to_write)
                        else:
                            data_to_write = [row[0], row[1], row[2], row[3], row[4], loss_respective_all[ix - 1]]
                            data_to_write_exit = [row[0], row[1], row[2], row[3], row[4], all_layers[ix - 1]]
                        rows_to_write.append(data_to_write)
                        rows_to_write_exit.append(data_to_write_exit)
                elif args.task_name == 'qnli':
                    for ix, row in enumerate(reader):
                        # 假设要写入第2列和第4列的数据，可以按如下方式选择这些数据：
                        if ix == 0:
                            data_to_write = [row[0], row[1], row[2], row[3]]  ##写入列名
                            data_to_write_exit = [row[0], row[1], row[2], row[3]]
                            print('row[0], row[1], row[2], row[3]: ', data_to_write)
                        else:
                            data_to_write = [row[0], row[1], row[2], loss_respective_all[ix - 1]]
                            data_to_write_exit = [row[0], row[1], row[2], all_layers[ix - 1]]
                        rows_to_write.append(data_to_write)
                        rows_to_write_exit.append(data_to_write_exit)

                elif args.task_name == 'mnli':
                    for ix, row in enumerate(reader):
                        # 假设要写入第2列和第4列的数据，可以按如下方式选择这些数据：
                        if ix == 0:
                            data_to_write = [row[0], row[1], row[2], row[3]]  ##写入列名
                            data_to_write_exit = [row[0], row[1], row[2], row[3]]
                            print('row[0], row[1], row[2], row[3]: ', data_to_write)
                        else:
                            data_to_write = [row[0], row[1], row[2], loss_respective_all[ix - 1]]
                            data_to_write_exit = [row[0], row[1], row[2], all_layers[ix - 1]]
                        rows_to_write.append(data_to_write)
                        rows_to_write_exit.append(data_to_write_exit)

                elif args.task_name == 'stackoverflow':
                    for ix, row in enumerate(reader):
                        # 假设要写入第2列和第4列的数据，可以按如下方式选择这些数据：
                        if ix == 0:
                            data_to_write = [row[0], row[1], row[2]]  ##写入列名
                            data_to_write_exit = [row[0], row[1], row[2]]
                            print('row[0], row[1], row[2]: ', data_to_write)
                        else:
                            data_to_write = [row[0], row[1], loss_respective_all[ix - 1]]
                            data_to_write_exit = [row[0], row[1], all_layers[ix - 1]]
                        rows_to_write.append(data_to_write)
                        rows_to_write_exit.append(data_to_write_exit)

                elif args.task_name == 'banking' or args.task_name == 'mcid':
                    for ix, row in enumerate(reader):
                        # 假设要写入第2列和第4列的数据，可以按如下方式选择这些数据：
                        if ix == 0:
                            data_to_write = [row[0], row[1], row[2]]  ##写入列名
                            data_to_write_exit = [row[0], row[1], row[2]]
                            print('row[0], row[1], row[2]: ', data_to_write)
                        else:
                            data_to_write = [row[0], row[1], loss_respective_all[ix - 1]]
                            data_to_write_exit = [row[0], row[1], all_layers[ix - 1]]
                        rows_to_write.append(data_to_write)
                        rows_to_write_exit.append(data_to_write_exit)

            with open(os.path.join(args.data_dir, 'dev_loss.tsv'), 'w', encoding="utf-8-sig",
                      newline='') as output_file:
                writer = csv.writer(output_file, delimiter='\t')
                for row in rows_to_write:
                    writer.writerow(row)

            with open(os.path.join(args.data_dir, 'dev_exit.tsv'), 'w', encoding="utf-8-sig",
                      newline='') as output_file:
                writer = csv.writer(output_file, delimiter='\t')
                for row in rows_to_write_exit:
                    writer.writerow(row)

        eval_loss = eval_loss / nb_eval_steps
        if args.output_mode == "classification":
            preds = np.argmax(preds, axis=1)
        elif args.output_mode == "regression":
            preds = np.squeeze(preds)

        print('eval_task: ',eval_task)

        if eval_task in ['stackoverflow', 'banking','mcid','rte','mrpc']:
            result = {"acc": (preds == out_label_ids).mean()}
        else:
            result = compute_metrics(eval_task, preds, out_label_ids)
        results.update(result)

        output_eval_file = os.path.join(eval_output_dir, prefix, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results {} *****".format(prefix))
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                print("  %s = %s" % (key, str(result[key])))
                writer.write("%s = %s\n" % (key, str(result[key])))

    if args.eval_all_checkpoints and patience != 0:
        if args.model_type == 'albert':
            model.albert.log_stats()
        elif args.model_type == 'bert':
            model.bert.log_stats()
        else:
            raise NotImplementedError()

    return results

# train_loss_all = []
def evaluate_train_loss(args, model, tokenizer, prefix="", patience=0):

    if args.model_type == 'albert':
        model.albert.set_regression_threshold(args.regression_threshold)
        model.albert.set_patience(patience)
        model.albert.reset_stats()
    elif args.model_type == 'bert':
        model.bert.set_regression_threshold(args.regression_threshold)
        model.bert.set_patience(patience)
        model.bert.reset_stats()
    else:
        raise NotImplementedError()

    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_task_names = ("mnli", "mnli-mm") if args.task_name == "mnli" else (args.task_name,)
    eval_outputs_dirs = (args.output_dir, args.output_dir + "-MM") if args.task_name == "mnli" else (args.output_dir,)

    results = {}
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        eval_dataset = load_and_cache_examples(args, eval_task, tokenizer, evaluate=False)

        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

        # multi-gpu eval
        if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
            model = torch.nn.DataParallel(model)

        # Eval!
        logger.info("***** Running evaluation on train_dataset_for_loss {} *****".format(prefix))
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        loss_respective_all = None
        all_layers = []
        labels_layer = [[] for _ in range(12)]
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3],"mode": args.mode}
                if args.model_type != "distilbert":
                    inputs["token_type_ids"] = (
                        batch[2] if args.model_type in ["bert", "xlnet", "albert"] else None
                    )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids
                outputs = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]

                loss_respective = outputs[2]

                if loss_respective_all is None:
                    loss_respective_all = loss_respective.tolist()
                else:
                    loss_respective_all += loss_respective.tolist()

                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs["labels"].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

            if args.mode == 1:  ##为了给出train_exit才用上这个
                labels_all = None  ###新添加，用来储存所有预测的labels

                for ix, logits_item in enumerate(outputs[3]):
                    if labels_all is None:
                        labels_all = logits_item.argmax(dim=1).unsqueeze(0)
                    else:
                        labels_all = torch.cat((labels_all, logits_item.argmax(dim=1).unsqueeze(0)), dim=0)

                for ix, logits_item in enumerate(outputs[3]):
                    labels_layer[ix].append(np.argmax(logits_item.detach().cpu().numpy(),axis=-1))


                ###关键之步
                # print('labels_all: ',labels_all)
                dim0, dim1 = labels_all.shape  ###没错，这里是基于记忆持续性
                #print('dim0,dim1:', dim0, dim1)  # dim是batchsize
                sample_layers = [12] * dim1  ###从1层开始，十二层为最后一层 普通bert
                for j in range(dim1):
                    for i in range(dim0):
                        label_single = labels_all[dim0 - i - 1][j]
                        if (label_single == inputs["labels"].detach().cpu().numpy()[j]):
                            sample_layers[j] = dim0 - i
                        else:
                            break  ###这里是基于持续性
                #print('sample_layers: ', sample_layers)
                """
                dim0, dim1 = labels_all.shape  ###没错，这里是基于记忆持续性
                sample_layers = [12] * dim1  ###从1层开始，十二层为最后一层 普通bert
                for j in range(dim1):
                    for i in range(dim0):
                        label_single = labels_all[dim0 - i - 1][j]
                        if (label_single == inputs["labels"].detach().cpu().numpy()[j]):
                            sample_layers[j] = dim0 - i
                """

                all_layers.append(sample_layers)

        if args.mode == 3:
            print('evaluate_loss_respective_all: ',loss_respective_all)



        if args.mode == 1 or args.mode == 3:  ##为了给出train_loss才用上这个
            all_layers = list(itertools.chain.from_iterable(all_layers))
            # print('len(loss_respective_all): ', len(loss_respective_all))
            print('loss_respective_all: ', loss_respective_all)
            print('all_layers: ', all_layers)
            #train_loss_all.append(loss_respective_all)
            for id,layer_label in enumerate(labels_layer):
                layer_label = list(itertools.chain.from_iterable(layer_label))
                print(id,"  acc: ", (layer_label == out_label_ids).mean())


            rows_to_write = []
            rows_to_write_exit = []
            with open(os.path.join(args.data_dir, "train.tsv"), 'r', encoding="utf-8-sig") as input_file:
                reader = list(csv.reader(input_file, delimiter="\t", quotechar=None))

                if args.task_name == 'sst-2':
                    for ix, row in enumerate(reader):
                        # 假设要写入第2列和第4列的数据，可以按如下方式选择这些数据：
                        if ix == 0:
                            data_to_write = [row[0], row[1]]  ##写入列名
                            data_to_write_exit = [row[0], row[1]]
                            print('row[1],row[2]: ', data_to_write)

                        else:
                            #print(ix - 1, " ", loss_respective_all[ix - 1])
                            data_to_write = [row[0], loss_respective_all[ix - 1]]  # 对于sst-2任务这种的写入方式  记得ix要减1位，对于
                            data_to_write_exit = [row[0], all_layers[ix - 1]]
                        rows_to_write.append(data_to_write)
                        rows_to_write_exit.append(data_to_write_exit)
                elif args.task_name == 'rte':
                    for ix, row in enumerate(reader):
                        # 假设要写入第2列和第4列的数据，可以按如下方式选择这些数据：
                        if ix == 0:
                            data_to_write = [row[0], row[1], row[2], row[3]]  ##写入列名
                            data_to_write_exit = [row[0], row[1], row[2], row[3]]
                            print('row[1],row[2]: ', data_to_write)
                        else:
                            #print(ix - 1, " ", loss_respective_all[ix - 1])
                            data_to_write = [row[0], row[1], row[2], loss_respective_all[ix - 1]]
                            data_to_write_exit = [row[0], row[1], row[2], all_layers[ix - 1]]
                        rows_to_write.append(data_to_write)
                        rows_to_write_exit.append(data_to_write_exit)
                elif args.task_name == 'mrpc':
                    for ix, row in enumerate(reader):
                        # 假设要写入第2列和第4列的数据，可以按如下方式选择这些数据：
                        if ix == 0:
                            data_to_write = [row[0], row[1], row[2], row[3], row[4]]  ##写入列名
                            data_to_write_exit = [row[0], row[1], row[2], row[3], row[4]]
                            print('row[1],row[2]: ', data_to_write)
                        else:
                            # print(ix,' len(row): ',len(row))
                            # print('row: ',row)
                            # print('len(loss_respective_all): ', len(loss_respective_all))
                            data_to_write = [row[1], row[2], row[3], row[4],
                                             loss_respective_all[ix - 1]]  # , row[2], row[3], row[4]]
                            data_to_write_exit = [row[1], row[2], row[3], row[4], all_layers[ix - 1]]
                        rows_to_write.append(data_to_write)
                        rows_to_write_exit.append(data_to_write_exit)

                elif args.task_name == 'qqp':
                    for ix, row in enumerate(reader):
                        # 假设要写入第2列和第4列的数据，可以按如下方式选择这些数据：
                        if ix == 0:
                            data_to_write = [row[0], row[1], row[2], row[3], row[4], row[5]]  ##写入列名
                            data_to_write_exit = [row[0], row[1], row[2], row[3], row[4], row[5]]
                            print('row[0], row[1], row[2], row[3], row[4], row[5]: ', data_to_write)
                        else:
                            data_to_write = [row[0], row[1], row[2], row[3], row[4], loss_respective_all[ix - 1]]
                            data_to_write_exit = [row[0], row[1], row[2], row[3], row[4], all_layers[ix - 1]]
                        rows_to_write.append(data_to_write)
                        rows_to_write_exit.append(data_to_write_exit)
                elif args.task_name == 'qnli':
                    for ix, row in enumerate(reader):
                        # 假设要写入第2列和第4列的数据，可以按如下方式选择这些数据：
                        if ix == 0:
                            data_to_write = [row[0], row[1], row[2], row[3]]  ##写入列名
                            data_to_write_exit = [row[0], row[1], row[2], row[3]]
                            print('row[0], row[1], row[2], row[3]: ', data_to_write)
                        else:
                            data_to_write = [row[0], row[1], row[2], loss_respective_all[ix - 1]]
                            data_to_write_exit = [row[0], row[1], row[2], all_layers[ix - 1]]
                        rows_to_write.append(data_to_write)
                        rows_to_write_exit.append(data_to_write_exit)

                elif args.task_name == 'mnli':
                    for ix, row in enumerate(reader):
                        # 假设要写入第2列和第4列的数据，可以按如下方式选择这些数据：
                        if ix == 0:
                            data_to_write = [row[0], row[1], row[2], row[3]]  ##写入列名
                            data_to_write_exit = [row[0], row[1], row[2], row[3]]
                            print('row[0], row[1], row[2], row[3]: ', data_to_write)
                        else:
                            data_to_write = [row[0], row[1], row[2], loss_respective_all[ix - 1]]
                            data_to_write_exit = [row[0], row[1], row[2], all_layers[ix - 1]]
                        rows_to_write.append(data_to_write)
                        rows_to_write_exit.append(data_to_write_exit)

                elif args.task_name == 'stackoverflow':
                    for ix, row in enumerate(reader):
                        # 假设要写入第2列和第4列的数据，可以按如下方式选择这些数据：
                        if ix == 0:
                            data_to_write = [row[0], row[1], row[2]]  ##写入列名
                            data_to_write_exit = [row[0], row[1], row[2]]
                            print('row[0], row[1], row[2]: ', data_to_write)
                        else:
                            data_to_write = [row[0], row[1], loss_respective_all[ix - 1]]
                            data_to_write_exit = [row[0], row[1], all_layers[ix - 1]]
                        rows_to_write.append(data_to_write)
                        rows_to_write_exit.append(data_to_write_exit)

                elif args.task_name == 'banking' or args.task_name == 'mcid':
                    for ix, row in enumerate(reader):
                        # 假设要写入第2列和第4列的数据，可以按如下方式选择这些数据：
                        if ix == 0:
                            data_to_write = [row[0], row[1], row[2]]  ##写入列名
                            data_to_write_exit = [row[0], row[1], row[2]]
                            print('row[0], row[1], row[2]: ', data_to_write)
                        else:
                            data_to_write = [row[0], row[1], loss_respective_all[ix - 1]]
                            data_to_write_exit = [row[0], row[1], all_layers[ix - 1]]
                        rows_to_write.append(data_to_write)
                        rows_to_write_exit.append(data_to_write_exit)

            with open(os.path.join(args.data_dir, 'train_loss.tsv'), 'w', encoding="utf-8-sig",
                      newline='') as output_file:
                writer = csv.writer(output_file, delimiter='\t')
                for row in rows_to_write:
                    writer.writerow(row)

            with open(os.path.join(args.data_dir, 'train_exit.tsv'), 'w', encoding="utf-8-sig",
                      newline='') as output_file:
                writer = csv.writer(output_file, delimiter='\t')
                for row in rows_to_write_exit:
                    writer.writerow(row)

        eval_loss = eval_loss / nb_eval_steps
        if args.output_mode == "classification":
            preds = np.argmax(preds, axis=1)
        elif args.output_mode == "regression":
            preds = np.squeeze(preds)
        if eval_task in ['stackoverflow', 'banking','mcid','rte','mrpc']:
            result = {"acc": (preds == out_label_ids).mean()}
        else:
            result = compute_metrics(eval_task, preds, out_label_ids)
        results.update(result)

        #stackoverflow
        # correct_list = [1 if p == t else 0 for p, t in zip(preds, out_label_ids)]
        # train_loss_all.append(correct_list)

    print('1111111111')
    if args.eval_all_checkpoints and patience != 0:
        print('2222222222')
        if args.model_type == 'albert':
            model.albert.log_stats()
        elif args.model_type == 'bert':
            model.bert.log_stats()
        else:
            raise NotImplementedError()

    return results




def load_and_cache_examples(args, task, tokenizer, evaluate=False):
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    processor = processors[task]()
    output_mode = output_modes[task]
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(
        args.data_dir,
        "cached_{}_{}_{}_{}".format(
            "dev" if evaluate else "train",
            list(filter(None, args.model_name_or_path.split("/"))).pop(),
            str(args.max_seq_length),
            str(task),
        ),
    )
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        label_list = processor.get_labels()
        if task in ["mnli", "mnli-mm"] and args.model_type in ["roberta", "xlmroberta"]:
            # HACK(label indices are swapped in RoBERTa pretrained model)
            label_list[1], label_list[2] = label_list[2], label_list[1]
        examples = (
            processor.get_dev_examples(args.data_dir) if evaluate else processor.get_train_examples(args.data_dir)
        )
        features = convert_examples_to_features(
            examples,
            tokenizer,
            label_list=label_list,
            max_length=args.max_seq_length,
            output_mode=output_mode,
            pad_on_left=bool(args.model_type in ["xlnet"]),  # pad on the left for xlnet
            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
            pad_token_segment_id=4 if args.model_type in ["xlnet"] else 0,
        )
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    if output_mode == "classification":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.float)

    if (args.mode == 2 or args.mode == 3) and not evaluate:

        with open(os.path.join(args.data_dir, "train_loss.tsv"), 'r', encoding="utf-8-sig", newline='') as input_file:
            reader = csv.reader(input_file, delimiter='\t')
            next(reader)  # 跳过第一行
            all_loss  = [float(row[-1]) for row in reader]

        all_loss = torch.tensor([f for f in all_loss], dtype=torch.float)
        print('all_loss.shape: ',all_loss.shape)

        topk_min = torch.topk(all_loss, k=5, largest=False).values[-1]
        topk_max = torch.topk(all_loss, k=5, largest=True).values[-1]

        # 剪切约束最大的五个值和最小的五个值到对应的范围内
        all_loss = torch.clamp(all_loss, topk_min, topk_max)
        # 进行最大最小值标准化
        normalized_loss = (all_loss - topk_min) / (topk_max - topk_min)

        #print('all_input_ids.shape: ',all_input_ids.shape)
        #print('all_attention_mask.shape: ', all_attention_mask.shape)
        #print('normalized_loss.shape: ', normalized_loss.shape)

        #normalized_loss = all_loss

        with open(os.path.join(args.data_dir, "train_exit.tsv"), 'r', encoding="utf-8-sig", newline='') as input_file:
            reader2 = csv.reader(input_file, delimiter='\t')
            next(reader2)  # 跳过第一行
            #all_exit  = [(int(row[-1])-1) for row in reader2]
            all_exit = [int(row[-1]) for row in reader2]

        all_exit = torch.tensor([f for f in all_exit], dtype=torch.long)

        """ 做专家数据的时候才要onehot，不是专家数据的时候不用
        exit_layer = torch.nn.functional.one_hot(all_exit)
        for item in exit_layer:
            for ix, i in enumerate(item):
                if i == 1:
                    item[ix:] = 1
                    break
        """
        exit_layer = all_exit
        dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels, normalized_loss, exit_layer)
        return dataset

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
    return dataset


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The input data dir. Should contain the .tsv files (or other data files) for the task.",
    )
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS),
    )
    parser.add_argument(
        "--task_name",
        default=None,
        type=str,
        required=True,
        help="The name of the task to train selected in the list: " + ", ".join(processors.keys()),
    )
    parser.add_argument(
        "--mode",
        default=1,
        type=int,
        required=False,
        help="mode=1，2 seperately represents the stage that training main model+classifiers and training Policy_network",
    )

    parser.add_argument("--alpha",
                        default=1,
                        type=float,
                        help="")

    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--patience",
        default='0',
        type=str,
        required=False,
    )


    parser.add_argument(
        "--regression_threshold",
        default=0,
        type=float,
        required=False,
    )
    # Other parameters
    parser.add_argument(
        "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument(
        "--evaluate_during_training", action="store_true", help="Run evaluation during training at each logging step.",
    )
    parser.add_argument(
        "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model.",
    )

    parser.add_argument(
        "--per_gpu_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.",
    )
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")

    parser.add_argument("--logging_steps", type=int, default=500, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory",
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets",
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
        "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")
    args = parser.parse_args()

    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    # Set seed
    set_seed(args)

    # Prepare GLUE task
    args.task_name = args.task_name.lower()
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))
    processor = processors[args.task_name]()
    args.output_mode = output_modes[args.task_name]
    label_list = processor.get_labels()
    num_labels = len(label_list)

    print("args.data_dir: ",args.data_dir)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=args.task_name,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    tokenizer = tokenizer_class.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    model = model_class.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)

    print('Total Model Parameters:', sum(param.numel() for param in model.parameters()))
    output_layers_param_num = sum(param.numel() for param in model.classifiers.parameters())
    print('Output Layers Parameters:', output_layers_param_num)
    single_output_layer_param_num = sum(param.numel() for param in model.classifiers[0].parameters())
    print('Added Output Layers Parameters:', output_layers_param_num - single_output_layer_param_num)

    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train and args.mode == 1:
        train_dataset = load_and_cache_examples(args, args.task_name, tokenizer, evaluate=False)
        global_step, tr_loss = train(args, train_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    elif args.do_train and args.mode == 2:
        print('args.mode: ', args.mode)
        train_dataset = load_and_cache_examples(args, args.task_name, tokenizer, evaluate=False)
        global_step, tr_loss = train_rl(args, train_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    elif args.do_train and args.mode == 3:
        print('args.mode: ', args.mode)
        train_dataset = load_and_cache_examples(args, args.task_name, tokenizer, evaluate=False)
        #global_step, tr_loss = train_alternate(args, train_dataset, model, tokenizer)
        global_step, tr_loss = train_both(args, train_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = (
            model.module if hasattr(model, "module") else model
        )  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

        # Load a trained model and vocabulary that you have fine-tuned
        model = model_class.from_pretrained(args.output_dir)
        tokenizer = tokenizer_class.from_pretrained(args.output_dir)
        model.to(args.device)

    # Evaluation
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        patience_list = [int(x) for x in args.patience.split(',')]
        tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
            )
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)

        for checkpoint in checkpoints:

            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""

            model = model_class.from_pretrained(checkpoint)
            model.to(args.device)

            print(f'Evaluation for checkpoint {prefix}')
            for patience in patience_list:

                result = evaluate(args, model, tokenizer, prefix=prefix, patience=patience)
                #evaluate_train_loss(args, model, tokenizer, patience=patience)

                result = dict((k + "_{}".format(global_step), v) for k, v in result.items())
                results.update(result)

        if args.mode == 1:
            print('Generating Training datasets loss and layer...')
            evaluate_train_loss(args, model, tokenizer)  #用于生成train_loss


    # loss_array = np.array(train_loss_all)
    #
    # # 定义 log2 阈值
    # log2_threshold = math.log(2)  # 如果 loss < log2(1.0)，则被认为是分类正确
    #
    # # 统计每个样本的分类正确又分类错误的次数
    # correct_and_incorrect_counts = {}
    # forget_list = []
    # num_evaluations = len(loss_array)
    #
    # for sample_index in range(loss_array.shape[1]):
    #     forget_times = 0
    #
    #     previous_result = 0
    #     result = 0
    #     for evaluation_index in range(num_evaluations):
    #         if loss_array[evaluation_index, sample_index] < log2_threshold: #对于实际预测，大于0.5说明预测正确
    #             result = 1
    #         else:
    #             result = 0
    #         if result == 0 and previous_result == 1:
    #             forget_times += 1
    #         previous_result = result
    #     if all(row[sample_index] > log2_threshold for row in loss_array) and forget_times == 0:
    #         forget_times = num_evaluations/2 + 1 #认为是最难的
    #
    #     correct_and_incorrect_counts[sample_index] = forget_times
    #     forget_list.append(forget_times)
    #
    # print('correct_and_incorrect_counts: ', correct_and_incorrect_counts)
    # for item, value in correct_and_incorrect_counts.items():
    #     if value > 0:
    #         print('item: ', item, ' value: ', value)
    # print('forget_list:　',forget_list)

    # with open('forget_list.csv', 'w', newline='') as csvfile:
    #     csv_writer = csv.writer(csvfile)
    #     csv_writer.writerows(forget_list)

    return results


if __name__ == "__main__":

    main()
