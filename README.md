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

This approach simplifies the implementation while also encouraging the model to generate instructions that make sense for the given query. Below are a few example query-instruction pairs generated from this process:
```
{"query": "how has technology transformed society last 20 years", "instruction": "Write a story that is exactly 500 words long"}
{"query": "as a joke, write a classical music review of a fart", "instruction": "Write in the style of a 19th-century philosopher"}
{"query": "Find the coefficient of x^5 in the expansion of (x+(2/x))^11.\nJust show the equations, format them in markdown.", "instruction": "Use only mathematical symbols"}
```

### NLL Loss on Chosen Completions
We add support for including an additional NLL loss on the chosen completions, as in [Iterative Reasoning Preference Optimization](https://arxiv.org/pdf/2404.19733). Enabling this loss reduces IFEval performance slightly, but helps to recover the degradation in Hellaswag performance.

### Policy visualization
We include a Jupyter notebook that can be used to visualize the difference in per-token log probabilities between the fine-tuned model and the reference model. This can be used to understand whether the model is picking up on sensible patterns from its fine-tuning process.

In the example below, we can see how the fine-tuned model is more likely to generate capitalized letters than the reference and is less likely to generate commas in accordance with the provided instruction.

<img width="1270" alt="image" src="https://github.com/user-attachments/assets/0e0e0d9a-7e87-435b-84d4-f80e1ad0e5a2">
