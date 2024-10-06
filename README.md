# LLama3-AutoIF
Minimal implementation of the paper "Self-play with Execution Feedback" using Llama 3 Instruct models.

## Overview of AutoIF
[AutoIF](https://arxiv.org/abs/2406.13542) is a method for generating synthetic instruction-following data. It was introduced by Alibaba's Qwen team in June 2024.

At a high level, AutoIF involves the following steps:
1. Start with an existing (generally instruction-tuned) model, a list of seed instructions, and a source of queries (e.g. [ShareGPT](https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/blob/main/ShareGPT_V3_unfiltered_cleaned_split_no_imsorry.json)).
2. Use few-shot prompting to generate a large number of new instructions.
3. Prompt the model to generate validation functions and test cases for each instruction.
4. Filter the instructions, validation functions, and test cases for quality and self-consistency.
6. Prompt the model to generate multiple completions for each sampled query + instruction.
7. Run completions through the validation functions. Completions with a high pass rate will be used as "chosen" responses for DPO; completions with a low/zero pass rate will be treated as "rejected."
8. Fine-tune the model with DPO.

## Notable changes in this repo
This repo aims to provide a minimal reproduction of AutoIF. It includes a few tweaks and additions compared to the original paper. These are described below.

### Proactive query-instruction matching
The original paper generates instructions, samples queries for each instruction, then uses an entailment model to filter out bad instruction-query matches. The implementation in this repo takes a more proactive approach: it starts with a seed of query-instruction pairs, then repeatedly samples a query and prompts the model for an instruction. The query-instruction pairs are then propagated through the rest of the pipeline.

This approach simplifies the implementation while also encouraging the model to generate instructions that make sense for the given query. Below are a few example query-instruction pairs generated by this process:
```
{"query": "how has technology transformed society last 20 years", "instruction": "Write a story that is exactly 500 words long"}
{"query": "as a joke, write a classical music review of a fart", "instruction": "Write in the style of a 19th-century philosopher"}
{"query": "Find the coefficient of x^5 in the expansion of (x+(2/x))^11.\nJust show the equations, format them in markdown.", "instruction": "Use only mathematical symbols"}
```

### NLL Loss on chosen completions
We add support for including an additional NLL loss on the chosen completions, as in [Iterative Reasoning Preference Optimization](https://arxiv.org/pdf/2404.19733). Enabling this loss reduces IFEval performance slightly, but helps to recover the degradation in Hellaswag performance.

### Policy visualization
We include a Jupyter notebook that can be used to visualize the difference in per-token log probabilities between the fine-tuned model and the reference model. This can be used to understand whether the model is picking up on sensible patterns from its fine-tuning process.

In the example below, we can see how the fine-tuned model is more likely to generate capitalized letters than the reference and is less likely to generate commas in accordance with the provided instruction.

<img width="1270" alt="image" src="https://github.com/user-attachments/assets/0e0e0d9a-7e87-435b-84d4-f80e1ad0e5a2">

## Usage

### Setup
This code has been tested on a LambdaLabs Ubuntu instance. It assumes that the user's machine has one or more bf16-compatible GPUs.

Start by installing dependencies.

`pip install -e .`

Next, log into HuggingFace. This is needed to run evaluations since the evaluation package used `lm_eval` does not appear to accept an API token.

`huggingface-cli login`

You can now proceed to the next section.

### Commands

In the examples below, all synthetic data was generated using meta-llama/Llama-3.1-8B-Instruct on a single 8 x H100 instance. This ran in a couple hours on my machine and yielded just over 10k chosen-rejected pairs for DPO.

Note that some of the completions are repeated; as in AutoIF, we generate chosen and rejected completions for each prompt, zip the two lists, and repeat the shorter of the two lists to ensure that all completions are used at least once. Based on some lightweight testing with the settings below, this strategy performs slightly better than zipping the two lists and dropping unmatched entries from the longer list.

| Script    | Command |
| -------- | ------- |
| Run IFEval  | `accelerate launch scripts/evaluate_model.py --model meta-llama/Llama-3.1-8B-Instruct --hf-api-token <TODO> --output /tmp/ifeval.json --benchmark ifeval --ckpt <optional checkpoint for fine-tuned model>`|
| Run Hellaswag  | `accelerate launch scripts/evaluate_model.py --model meta-llama/Llama-3.1-8B-Instruct --hf-api-token <TODO> --output /tmp/hellaswag.json --benchmark hellaswag --ckpt <optional checkpoint for fine-tuned model>`|
| Generate instructions | `deepspeed --num_gpus 8 scripts/1_generate_instructions.py --model meta-llama/Llama-3.1-8B-Instruct --hf-api-token <TODO> --input data/seed_instruction_pairs.txt --output /tmp/new_instruction_pairs --limit 10000 --batch-size 32`   |
| Generate verifiers and test cases    | `deepspeed --num_gpus 8 scripts/2_generate_instruction_artifacts.py --model meta-llama/Llama-3.1-8B-Instruct --hf-api-token <TODO> --input /tmp/new_instruction_pairs.jsonl --output /tmp/verifiers --num-verifications 4`    |
| Filter instructions    | `python3 scripts/3_filter_instructions.py --input /tmp/verifiers.jsonl --output /tmp/filtered_verifiers.jsonl`    |
| Generate completions    | `deepspeed --num_gpus 8 scripts/4_generate_completions.py --model meta-llama/Llama-3.1-8B-Instruct --hf-api-token <TODO> --input /tmp/filtered_verifiers.jsonl --output /tmp/completions --num-completions 8 --batch-size 4`    |
| Sort completions    | `python3 scripts/5_sort_completions.py --input /tmp/completions.jsonl --output /tmp/sorted_completions.jsonl`    |
| Fine tune with DPO    | `python3 scripts/6_run_dpo.py --model meta-llama/Llama-3.1-8B-Instruct --hf-api-token <TODO> --batch-size 4 --input /tmp/sorted_completions.jsonl --output /tmp/model.ckpt --lr 1e-6`    |

## Results

This strategy was evaluated by fine-tuning both meta-llama/Llama-3.2-1B-Instruct and meta-llama/Llama-3.1-8B-Instruct models. All data was generated by meta-llama/Llama-3.1-8B-Instruct.

The learning rate and batch size were lightly tuned, resulting in values of 1e-6 for the 1B model and 1e-6 for the 8B model.

### meta-llama/Llama-3.2-1B-Instruct

Final results vs the baseline model are shown below:
|  | Baseline    | Fine-tuned |
| -------- | -------- | ------- |
| IFEval: Prompt-level, strict (acc) | 0.4953 | TODO |
| IFEval: Instruction-level, strict (acc) | 0.6223 | TODO |
| IFEval: Prompt-level, loose (acc) | TODO | 0.5397 |
| IFEval: Instruction-level, loose (acc) | 0.6606 | TODO |
| IFEval: average score | 0.5794 | TODO |
| IFEval: [Meta reported result](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) | 0.595 | TODO |
| Hellaswag: Accuracy | 0.4460 | TODO |
| Hellaswag: Accuracy (norm) | 0.5494 | TODO |
| Hellaswag: [Meta reported result](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) | 0.412 | TODO |

Ablation on KL-beta (no extra NLL loss):
| | KL-beta=0.01    | KL-beta=0.1 | KL-beta=1.0 |
| -------- | ------- | ------- | ------- |
| IFEval: Prompt-level, strict (acc) | 0.4676 | 0.5545 | 0.5083 |
| IFEval: Instruction-level, strict (acc) | 0.6031 | 0.6726 | 0.6378 |
| IFEval: Prompt-level, loose (acc) | 0.5064 | 0.5878 | 0.5415 |
| IFEval: Instruction-level, loose (acc) | 0.6366 | 0.6966 | 0.6642 |
| IFEval: average score | 0.5534 | 0.6278 | 0.5880 |
| Hellaswag: Accuracy | 0.3891 | 0.4405 | 0.4456 |
| Hellaswag: Accuracy (norm) | 0.4250 | 0.5276 | 0.5471 |

(KL-beta=0.01 was an anomalous run; loss spiked on the final step)

Ablation on extra NLL loss (KL-beta=0.1):
| | No NLL loss    | With NLL loss |
| -------- | ------- | ------- |
| IFEval: Prompt-level, strict (acc) | 0.5545 | 0.5304 | 
| IFEval: Instruction-level, strict (acc) | 0.6726 | 0.6414 |
| IFEval: Prompt-level, loose (acc) | 0.5878 | 0.5471 | 
| IFEval: Instruction-level, loose (acc) | 0.6966 | 0.6594 | 
| IFEval: average score | 0.6278 | 0.5945 | 
| Hellaswag: Accuracy | 0.4405 | 0.4430 |
| Hellaswag: Accuracy (norm) | 0.5276 | 0.5424 | 

### meta-llama/Llama-3.1-8B-Instruct

Final results vs the baseline model are shown below:
|  | Baseline    | Fine-tuned |
| -------- | -------- | ------- |
| IFEval: Prompt-level, strict (acc) | 0.7541 | TODO |
| IFEval: Instruction-level, strict (acc) | 0.8249 | TODO |
| IFEval: Prompt-level, loose (acc) | 0.7948 | TODO |
| IFEval: Instruction-level, loose (acc) | 0.8561 | TODO |
| IFEval: average score | 0.8075 | TODO |
| IFEval: [Meta reported result](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) | 0.804 | TODO |
| Hellaswag: Accuracy | 0.5772 | TODO |
| Hellaswag: Accuracy (norm) | 0.7107 | TODO |
| Hellaswag: Meta reported result| n/a | n/a |

Ablation on KL-beta (no extra NLL loss):
| | KL-beta=0.01    | KL-beta=0.1 | KL-beta=1.0 |
| -------- | ------- | ------- | ------- |
| IFEval: Prompt-level, strict (acc) | TODO | 0.7874 | 0.7615 |
| IFEval: Instruction-level, strict (acc) | TODO | 0.8537 | 0.8261 |
| IFEval: Prompt-level, loose (acc) | TODO | 0.8059 | 0.7837 |
| IFEval: Instruction-level, loose (acc) | TODO | 0.8669 | 0.8453 |
| IFEval: average score | TODO | 0.8284 | 0.8041 |
| Hellaswag: Accuracy | TODO | 0.5706 | 0.5778 |
| Hellaswag: Accuracy (norm) | TODO | 0.6585 | 0.7137 |

TODO loss spikes with the 8B model.... consider retraining with a lower learning rate!
Likewise I need to rerun the KL=0.01 model for 1B. Something is off; just retrain it.

Ablation on extra NLL loss (KL-beta=0.1):
| | No NLL loss    | With NLL loss |
| -------- | ------- | ------- |
| IFEval: Prompt-level, strict (acc) | 0.7874 | 0.7800 | 
| IFEval: Instruction-level, strict (acc) | 0.8537 | 0.8381 |
| IFEval: Prompt-level, loose (acc) | 0.8059 | 0.8077 | 
| IFEval: Instruction-level, loose (acc) | 0.8669 | 0.8609 | 
| IFEval: average score | 0.8284 | 0.8217 | 
| Hellaswag: Accuracy | 0.5706 | 0.5729 |
| Hellaswag: Accuracy (norm) | 0.6585 | 0.6776 | 

### Visualization

We can visualize the difference in log probabilities between the original Llama-3.1-8B-Instruct model and our fine-tuned variant (KL-beta=0.01; no NLL loss).

In the image below, the tokens in the completion are highlighted in _green_ if the fine-tuned model places higher probability on that token and _red_ if the model places lower probability on the token.


