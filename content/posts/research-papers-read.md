---
title: "Paper Read"
date: 2023-09-23
author: Nitin
tags: [papers]
---

The blog contains a summary of each research paper I have recently read and found interesting.


## LLM as optimisers (Google)
They propose Optimization by PROmpting (OPRO), a simple and effective approach to leverage large language models (LLMs) as optimizers, where the optimization task is described in natural language. They show how linear regression and travelling salesman problem can be framed using text and the LLM does a decent job at arriving at the solution. But the most interesting part for me is covered under a section on prompt optimisation. Optimising the prompt i.e, finding the best prompt to get the desired response would be a boon for prompt engineers.
    1. Paper - https://arxiv.org/abs/2309.03409
    2. Code - Not mentioned
    3. Benchmark data - GSM8k and BigBench Hard


## Generative Agents: Interactive Simulacra of Human Behavior (Google)
They create 25 agents in a sandbox environment, akins to The Sims game, who interact with each other, simulating a real world interactions. And it all powered by gpt3.5-turbo. One aspect that stands out is the agent architecture which is designed to pick relevant pieces from the agent’s memory (which can be very long) and use them to inform agent behavior. I think this could be a solution (or at least offer some ideas) for the maintaining the relevant chat context problem for multi-turn conversations.
    1. Paper - https://arxiv.org/abs/2304.03442
    2. Code - https://github.com/joonspk-research/generative_agents
    3. Benchmark data - NA


## From Sparse to Dense: GPT-4 Summarization with Chain of Density Prompting (Multiple)
From Sparse to Dense: GPT-4 Summarization with Chain of Density Prompting. They talk about packing more information into the summary by using this Chain of Density approach - recursively add entities without increasing summary length. Density is entities per token across the whole summary. Interesting prompting and evaluation methodology that could be useful for different summarisation requirements.
    1. Paper - https://arxiv.org/pdf/2309.04269
    2. Code - Not available
    3. Benchmark data - https://huggingface.co/datasets/griffin/chain_of_density


## Chain-Of-VErification reduces hallucination in large language models (Meta)
Paper that describes a methodology for reducing hallucinations in LLM models - specifically LLAMA 2 models. Reduces longform hallucinations via LLM double-checking its own work with shortform questions. The Chain-of-Verification (CoVe) recipe:
    1. Generate baseline response
    2. Generate a verification plan (set of questions)
    3. Execute plan (answer questions)
    4. Generate Final Verified Response (using answers)
Could be a useful recipe for reducing, if not eliminating hallucination, for other LLM models.
    1. Paper - https://arxiv.org/pdf/2309.11495.pdf
    2. Code - Not available
    3. Benchmark data - WIKIDATA, WIKI-CATEGORY LIST, MULTISPANQA

## Think before you speak: Training Language ModelsWith Pause Tokens (Google)
In this study they try to use special `<pause>`tokens at the time of both training and inference. The idea is to use those pause tokens as a way to delay the response generation from the LLM model - allowing it more time to think and hence generate better responses. They empirically evaluate pause-training on decoder-only models of 1B and 130M parameters with causal pretraining on C4, and on downstream tasks covering reasoning, question-answering, general understanding and fact recall. They see good improvement. Interesting idea for future.
    1. Paper - https://arxiv.org/pdf/2310.02226.pdf
    2. Code - Not available
    3. Benchmark data - GSM8k, SQuAD, CoQA etc