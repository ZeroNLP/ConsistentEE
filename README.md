# ConsistentEE: A Consistent and Hardness-Guided Early Exiting Method for Accelerating Language Models Inference (AAAI 2024)
Code for the paper titled "ConsistentEE: A Consistent and Hardness-Guided Early Exiting Method for Accelerating Language Models Inference" [AAAI 2024 Main Track] 

**Due to this is a new area about Large Language Model's inference accerleration, we are open to any advice for improving our work.**

<p align="center">
<img width="1394" src="https://github.com/yihuaihong/yihuaihong.github.io/blob/main/images/Main%20Structure.png">
</p>

[**ConsistentEE: A Consistent and Hardness-Guided Early Exiting Method for Accelerating Language Models Inference**](https://arxiv.org/abs/2312.11882)      
[Ziqian Zeng](https://ziqianzeng.github.io/)$^\*$, 
[Yihuai Hong](https://yihuaihong.github.io/)$^\*$,[Hongliang Dai](https://hldai.github.io/),
Huiping Zhuang,
Cen Chen<br/>
\* equal contribution 

- We propose **ConsistentEE**, an early exiting method that can achieve consistency during training and inference by formulating the early exiting problem as a reinforcement learning problem.
- We propose a concept named Memorized Layer to measure the hardness of an instance. We incorporate it into the reward function to allow an instance to balance the accuracy and acceleration - depending on individual hardness.
- The experimental results show that our method can outperform other baselines on natural language understanding and generation tasks.

## Requirements
Install the necessary packages with: 
```
$ pip install -r requirements.txt
```

## Experiments
On encode-only models, we experimented with six tasks in GLUE, the MCID task and StackOverflow task.    
On decode-only models, we experimented with Alpaca/Dolly dataset and CNN/DM dataset.     


<!-- Please see the [scripts](scripts/) and run shell files to train or evaluate on each dataset.    
```bash
$ python run_[TASK_NAME]_[DATASET_NAME].sh
```  -->


## BibTeX
If you find this repo useful for your research, please consider citing our paper:

```
@misc{zeng2023consistentee,
      title={ConsistentEE: A Consistent and Hardness-Guided Early Exiting Method for Accelerating Language Models Inference}, 
      author={Ziqian Zeng and Yihuai Hong and Hongliang Dai and Huiping Zhuang and Cen Chen},
      year={2023},
      eprint={2312.11882},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
