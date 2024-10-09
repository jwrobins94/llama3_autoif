# LLama3-AutoIF
Self-contained implementation of the paper [Self-play with Execution Feedback](https://arxiv.org/pdf/2406.13542v3) using Llama 3 Instruct models.

Reproducibility notes:
1. All experiments were run on LambdaLabs using a single 8 x H100 instance.
2. Dataset generation runs in a few hours for ~$100 in total cost. The data is available on HuggingFace [here](https://huggingface.co/datasets/jwrobins94/llama3-autoif/tree/main).
3. Using the provided code and LLama 3.1 8B for data generation results in ~15k chosen-rejected pairs. Not all completions are unique (see "Commands").
4. Fine-tuning for 1 epoch takes ~15 minutes for Llama 3.1 8B.

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

## Setup
This code has been tested on a LambdaLabs Ubuntu instance. It assumes that the user's machine has one or more bf16-compatible GPUs.

Start by installing dependencies.

`pip install -e .`

Next, log into HuggingFace. This is needed to run evaluations since the evaluation package used (`lm_eval`) does not appear to accept an API token.

`huggingface-cli login`

## Experiments: 2024-10-08

### Changes
This change introduces several data quality improvements, including:
1. Seed instructions: better match the style between the query and the instruction so that the model will do the same when few-shot prompted.
2. Queries: switch from ShareGPT to Anthropic's helpful-base. This results in more readable, self-contained queries.
4. Instructions: prompt the model to think-step-by-step in comments and to write its code such that multiple interpretations of the instruction are accepted.
5. Instructions: increase max generation length to 1024
6. Completions: improve prompt specificity and provide a sampled verification function in the prompt
7. Completions: increase max generation length to 1024

These changes substantially increase data yield by improving the completion pass:fail rate from 1:3 to approximately 1:1:
1. The previous version began with 10k queries and resulted in 1836 unique instances, 14270 unique completions, and 10577 chosen-rejected pairs.
2. The updated version begins with 10k queries and results in 2781 unique instances, 21322 unique completions, and 15760 chosen-rejected pairs.

Below are a few example instructions created from this process:
```
{
  "query": "My significant other recently moved across the country for work. He won't be back for at least another year. Do you have any advice on how to have a successful long-distance relationship?",
  "instruction": "Be at least 5 sentences long but no longer than 10 sentences."
}
{
  "query": "What is minestrone?",
  "instruction": "Answer with a single sentence that is exactly 15 words long and starts with the letter \"M\"."
}
{
  "query": "I'm travelling overseas and need to learn how to say \"I'm allergic\" in a few languages. Can you help me?",
  "instruction": "Answer with a JSON object where keys are language codes and values are phrases."
}
{
  "query": "What are some good Ecuadorian meals?",
  "instruction": "List at least 3 dishes with exactly 3 ingredients each in alphabetical order, in a comma-separated string."
}
{
  "query": "how can i make a doll house for a boy?",
  "instruction": "use a numbered list with exactly 5 items."
}
```

Below is an example verification function for the instruction "Answer with exactly 3 sentences, each starting with a different letter of the alphabet.":
```
def evaluate(response: str) -> bool:  # Function to evaluate if the response follows the instruction
    # First, split the response into sentences
    sentences = response.split('. ')  # Assuming the response is in the format'sentence. sentence. sentence'
    
    # Check if the number of sentences is 3
    if len(sentences)!= 3:
        return False  # If not 3 sentences, immediately return False

    # Initialize a set to store the first letters of each sentence
    first_letters = set()
    
    # Get the first letter of each sentence and add it to the set
    for sentence in sentences:
        # Strip any leading or trailing whitespace from the sentence
        sentence = sentence.strip()
        # Get the first character of the sentence (assuming the sentence is not empty)
        if sentence:  # Check if the sentence is not empty
            first_letter = sentence[0].upper()  # Convert the letter to uppercase
            # Add the first letter to the set
            first_letters.add(first_letter)

    # Check if all letters are unique and in the alphabet (A to Z)
    # If any letter is repeated or not in the alphabet, return False
    if len(first_letters)!= 3 or not first_letters.issubset('ABCDEFGHIJKLMNOPQRSTUVWXYZ'):
        return False

    # If all checks pass, the response follows the instruction, so return True
    return True
```

### Commands

Results can be reproduced using the following commands:
| Script    | Command |
| -------- | ------- |
| Run IFEval  |  no change |
| Run Hellaswag  | no change |
| Generate instructions | `deepspeed --num_gpus 8 scripts/1_generate_instructions.py --model meta-llama/Llama-3.1-8B-Instruct --hf-api-token <TODO> --input data/seed_instruction_pairs_hh.txt --output /tmp/new_instruction_pairs --limit 10000 --batch-size 32` |
| Generate verifiers and test cases    | no change |
| Filter instructions    | no change |
| Generate completions    | no change |
| Sort completions    | no change |
| Fine tune with DPO    | `python3 scripts/6_run_dpo.py --model meta-llama/Llama-3.2-1B-Instruct --hf-api-token <TODO> --batch-size 8 --grad-acc-steps 4 --lr 3e-6 --input /tmp/sorted_completions.jsonl --output /tmp/model.ckpt`  |

See "Experiments: 2024-10-06" for the base commands.

### Results

With light tuning of the learning rate (up to 3e-6) and batch size (up to 256), performance on both IFEval and Hellaswag are also improved. Results for meta-llama/Llama-3.2-1B-Instruct are shown below:
|  | Baseline    | Fine-tuned (old) | Fine-tuned (new) |
| -------- | -------- | ------- | ------- |
| IFEval: Prompt-level, strict (acc) | 0.4953 | 0.5545 | 0.5693 |
| IFEval: Instruction-level, strict (acc) | 0.6223 | 0.6726 | 0.6894 |
| IFEval: Prompt-level, loose (acc) | 0.5397 | 0.5878| 0.6025 |
| IFEval: Instruction-level, loose (acc) | 0.6618 | 0.6966 | 0.7146 |
| IFEval: average score | 0.5797 | 0.6278 | 0.6440 |
| IFEval: [Meta reported result](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) | 0.595 | n/a | n/a |
| Hellaswag: Accuracy | 0.4460 | 0.4405 | 0.4462 |
| Hellaswag: Accuracy (norm) | 0.5494 | 0.5276 | 0.5371 |
| Hellaswag: [Meta reported result](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) | 0.412 | n/a | n/a |

The new data can be found on HuggingFace under [jwrobins94/llama3-autoif-v2](https://huggingface.co/datasets/jwrobins94/llama3-autoif-v2/tree/main).

## Experiments: 2024-10-06

**These results are outdated and are retained for completeness. See "Experiments: 2024-10-08" for the latest results on the newer dataset.**

### Changes
The original paper generates instructions, samples queries for each instruction, then uses an entailment model to filter out bad instruction-query matches. The implementation in this repo takes a more proactive approach: it starts with a seed of query-instruction pairs, then repeatedly samples a query and prompts the model for a corresponding instruction. The query-instruction pairs are then propagated through the rest of the pipeline.

This approach simplifies the implementation while also encouraging the model to generate instructions that make sense for the given query. Below are a few example query-instruction pairs generated by this process:
```
{"query": "how has technology transformed society last 20 years", "instruction": "Write a story that is exactly 500 words long"}
{"query": "as a joke, write a classical music review of a fart", "instruction": "Write in the style of a 19th-century philosopher"}
{"query": "Find the coefficient of x^5 in the expansion of (x+(2/x))^11.\nJust show the equations, format them in markdown.", "instruction": "Use only mathematical symbols"}
```

We also add support for including an additional NLL loss on the chosen completions, as in [Iterative Reasoning Preference Optimization](https://arxiv.org/pdf/2404.19733). Enabling this loss reduces IFEval performance slightly, but helps to recover the degradation in Hellaswag performance.

### Commands

In the examples below, all synthetic data was generated using meta-llama/Llama-3.1-8B-Instruct on a single 8 x H100 instance. This ran in a couple hours on my machine and yielded just over 10k chosen-rejected pairs for DPO.

Note that some of the completions are repeated. As in AutoIF, we generate chosen and rejected completions for each prompt, zip the two lists, and repeat the shorter of the two lists to ensure that all completions are used at least once. Based on some lightweight testing with the settings below, this strategy performs slightly better than zipping the two lists and dropping unmatched entries from the longer list.

| Script    | Command |
| -------- | ------- |
| Run IFEval  | `accelerate launch scripts/evaluate_model.py --model meta-llama/Llama-3.1-8B-Instruct --hf-api-token <TODO> --output /tmp/ifeval.json --benchmark ifeval --ckpt <optional checkpoint for fine-tuned model>`|
| Run Hellaswag  | `accelerate launch scripts/evaluate_model.py --model meta-llama/Llama-3.1-8B-Instruct --hf-api-token <TODO> --output /tmp/hellaswag.json --benchmark hellaswag --ckpt <optional checkpoint for fine-tuned model>`|
| Generate instructions | `deepspeed --num_gpus 8 scripts/1_generate_instructions.py --model meta-llama/Llama-3.1-8B-Instruct --hf-api-token <TODO> --input data/seed_instruction_pairs_sharegpt.txt --output /tmp/new_instruction_pairs --limit 10000 --batch-size 32`   |
| Generate verifiers and test cases    | `deepspeed --num_gpus 8 scripts/2_generate_instruction_artifacts.py --model meta-llama/Llama-3.1-8B-Instruct --hf-api-token <TODO> --input /tmp/new_instruction_pairs.jsonl --output /tmp/verifiers --num-verifications 4`    |
| Filter instructions    | `python3 scripts/3_filter_instructions.py --input /tmp/verifiers.jsonl --output /tmp/filtered_verifiers.jsonl`    |
| Generate completions    | `deepspeed --num_gpus 8 scripts/4_generate_completions.py --model meta-llama/Llama-3.1-8B-Instruct --hf-api-token <TODO> --input /tmp/filtered_verifiers.jsonl --output /tmp/completions --num-completions 8 --batch-size 4`    |
| Sort completions    | `python3 scripts/5_sort_completions.py --input /tmp/completions.jsonl --output /tmp/sorted_completions.jsonl`    |
| Fine tune with DPO    | `python3 scripts/6_run_dpo.py --model meta-llama/Llama-3.1-8B-Instruct --hf-api-token <TODO> --batch-size 4 --input /tmp/sorted_completions.jsonl --output /tmp/model.ckpt --lr 1e-6`    |

### Results

This strategy was evaluated by fine-tuning both meta-llama/Llama-3.2-1B-Instruct and meta-llama/Llama-3.1-8B-Instruct models. All data was generated by meta-llama/Llama-3.1-8B-Instruct.

The data is available on HuggingFace [here](https://huggingface.co/datasets/jwrobins94/llama3-autoif/tree/main).

The learning rate and batch size were lightly tuned, resulting in values of 1e-6 and 32 (4 per GPU) for both models.

#### meta-llama/Llama-3.2-1B-Instruct

Final results vs the baseline model are shown below:
|  | Baseline    | Fine-tuned (KL-beta=0.1) |
| -------- | -------- | ------- |
| IFEval: Prompt-level, strict (acc) | 0.4953 | 0.5545 |
| IFEval: Instruction-level, strict (acc) | 0.6223 | 0.6726 |
| IFEval: Prompt-level, loose (acc) | 0.5397 | 0.5878|
| IFEval: Instruction-level, loose (acc) | 0.6618 | 0.6966 |
| IFEval: average score | 0.5797 | 0.6278 |
| IFEval: [Meta reported result](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) | 0.595 | n/a |
| Hellaswag: Accuracy | 0.4460 | 0.4405 |
| Hellaswag: Accuracy (norm) | 0.5494 | 0.5276 |
| Hellaswag: [Meta reported result](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) | 0.412 | n/a |

Ablation on KL-beta (no extra NLL loss):
| | KL-beta=0.1 | KL-beta=1.0 |
| -------- | ------- | ------- |
| IFEval: Prompt-level, strict (acc) | 0.5545 | 0.5083 |
| IFEval: Instruction-level, strict (acc) | 0.6726 | 0.6378 |
| IFEval: Prompt-level, loose (acc) | 0.5878 | 0.5415 |
| IFEval: Instruction-level, loose (acc) | 0.6966 | 0.6642 |
| IFEval: average score | 0.6278 | 0.5880 |
| Hellaswag: Accuracy | 0.4405 | 0.4456 |
| Hellaswag: Accuracy (norm) | 0.5276 | 0.5471 |

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

#### meta-llama/Llama-3.1-8B-Instruct

Final results vs the baseline model are shown below (KL-beta=0.1):
|  | Baseline    | Fine-tuned |
| -------- | -------- | ------- |
| IFEval: Prompt-level, strict (acc) | 0.7541 | 0.7874 |
| IFEval: Instruction-level, strict (acc) | 0.8249 | 0.8537 |
| IFEval: Prompt-level, loose (acc) | 0.7948 | 0.8059 |
| IFEval: Instruction-level, loose (acc) | 0.8561 | 0.8669 |
| IFEval: average score | 0.8075 | 0.8284 |
| IFEval: [Meta reported result](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) | 0.804 | n/a |
| Hellaswag: Accuracy | 0.5772 | 0.5706 |
| Hellaswag: Accuracy (norm) | 0.7107 | 0.6585 |

Ablation on KL-beta (no extra NLL loss):
| | KL-beta=0.1 | KL-beta=1.0 |
| ------- | ------- | ------- |
| IFEval: Prompt-level, strict (acc) | 0.7874 | 0.7615 |
| IFEval: Instruction-level, strict (acc) | 0.8537 | 0.8261 |
| IFEval: Prompt-level, loose (acc) | 0.8059 | 0.7837 |
| IFEval: Instruction-level, loose (acc) | 0.8669 | 0.8453 |
| IFEval: average score | 0.8284 | 0.8041 |
| Hellaswag: Accuracy | 0.5706 | 0.5778 |
| Hellaswag: Accuracy (norm) | 0.6585 | 0.7137 |

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

#### Visualization

We can visualize the difference in log probabilities between the original Llama-3.1-8B-Instruct model and our fine-tuned variant (KL-beta=0.1).

In the images below, the tokens in the completion are highlighted in _green_ if the fine-tuned model places higher probability on that token and _red_ if the model places lower probability on the token.

<img width="1241" alt="image" src="https://github.com/user-attachments/assets/a751ecd6-0639-4109-bbc3-9f6cb66b6912">

<img width="1275" alt="image" src="https://github.com/user-attachments/assets/e1ab57b5-a1b6-4a07-846e-59424b5cf0a3">

<img width="1274" alt="image" src="https://github.com/user-attachments/assets/2767cf76-bf93-49b6-b118-34087797709b">


