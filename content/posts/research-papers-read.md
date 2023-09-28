---
title: "Interesting research papers in generative ai space"
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