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

## Llemma: An Open Language Model For Mathematics (Eulether)
7B and 34B parameter llama models trained on Proof-Pile-2 data, a mixture of scientific papers, web data
containing mathematics, and mathematical code, yielding LLEMMA. On the MATH benchmark LLEMMA outperforms all known open base models, as well as the unreleased Minerva model suite on an equi-parameter basis. Moreover, LLEMMA is capable of tool use and formal theorem proving without any further finetuning.
    1. Paper - https://arxiv.org/pdf/2310.10631.pdf
    2. Code - https://github.com/EleutherAI/math-lm
    3. Benchmark data - Proof-Pile-2, https://github.com/EleutherAI/math-lm/tree/main/proof_pile_2

## Self-RAG: Learning to Retrieve, Generate and Critique through Self-Reflection (University of Washington, IBM AI Research, Allen Institute for A)
Self-RAG is a new framework that trains and controls an arbitrary LLM through Self-reflection tokens. In particular, at every segment (e.g., sentence), Self-RAG can:
    * Retrieve: Self-RAG first decodes a retrieval token to evaluate the utility of retrieval and control a retrieval component. If retrieval is required, our LM calls an external retrieval module to find top relevant documents, using input query and previous generation.
    * Generate: If retrieval is not required, the model predicts the next output segment, as it does in a standard LM. If retrieval is needed, the model first generates generates critique token evaluating whether retrieved documents are relevant, and then generate continuation conditioned on the retrieved passages.
    * Critique: If retrieval is required, the model further evaluates if passages support generation. Finally, a new critique token evaluates the overall utility of the response.

Self-RAG outperforms vanilla ChatGPT or LLama2-chat across six tasks, and outperforms those SOTA models with widely-used retrieval-augmentation methods in most tasks by large margin
    1. Paper - https://arxiv.org/abs/2310.11511
    2. Code - https://github.com/AkariAsai/self-rag
    3. Data - https://drive.google.com/drive/folders/18na_ayid-8NjPOd18vpDx8iBoyT3akSL?usp=share_link
    4. Website - https://selfrag.github.io/

## Matryoshka Diffusion Models

## Phind models

## Chain-of-Table: Evolving Tables in the Reasoning Chain for Table Understanding:
They conduct step-by-step reasoning as step-by-step tabular operations to form a chain of tables. The
tables in the chain are the transformed tables by the tabular operations, representing the intermediate reasoning results. This procedure resemblesthe thought of reasoning in Chain-of-Thought. Achieves new state-of-the-art performance on WikiTQ, FeTaQA, and TabFact benchmarks across multiple LLM choices
    1. Paper - https://arxiv.org/pdf/2401.04398.pdf
    2. Code - https://github.com/wenhuchen/TableCoT/

![](/img/Chain_of_tables.png "Chain of Tables Illustration")

## Blending Is All You Need
This paper and the code don't look fully developed. They talk about randomly using an LLM (from a group of LLMs) per token, and that apprently improves the generation. Not many people in reddit etc are fully convinced

## Mixtral MoE
I was more curious about the MoE approach of creating sparse models. The huggingface blog does a better job - https://huggingface.co/blog/moe

To recap, in MoEs we replace every FFN layer of the transformer model with an MoE layer, which is composed of a gate network and a certain number of experts

![](/img/00_switch_transformer.png "MoE layer from the [Switch Transformers paper](https://arxiv.org/abs/2101.03961)")

Although MoEs provide benefits like efficient pretraining and faster inference compared to dense models, they also come with challenges:

Training: MoEs enable significantly more compute-efficient pretraining, but they’ve historically struggled to generalize during fine-tuning, leading to overfitting.
Inference: Although a MoE might have many parameters, only some of them are used during inference. This leads to much faster inference compared to a dense model with the same number of parameters. However, all parameters need to be loaded in RAM, so memory requirements are high. For example, given a MoE like Mixtral 8x7B, we’ll need to have enough VRAM to hold a dense 47B parameter model. Why 47B parameters and not 8 x 7B = 56B? That’s because in MoE models, only the FFN layers are treated as individual experts, and the rest of the model parameters are shared. At the same time, assuming just two experts are being used per token, the inference speed (FLOPs) is like using a 12B model (as opposed to a 14B model), because it computes 2x7B matrix multiplications, but with some layers shared (more on this soon).

## Direct Preference Optimisation
The paper is a long read. This video explains it succintly - https://www.youtube.com/watch?v=XZLc09hkMwA

To recap, DPO is able to bypass both fitting an explicit reward and performing RL to learn the policy using
a single maximum likelihood objective.

Huggingface article about DPO and other alternatives:
https://huggingface.co/blog/pref-tuning

## Mamba - better than transformers?

This paper proposes and alternative architecture based on the State Space Models. This discussion on reddit captures the essence of Mamba - https://www.reddit.com/r/MachineLearning/comments/190q1vb/d_so_mamba_vs_transformers_is_the_hype_real/

How it is different from RNNS:
* Mamba has a linear activation function between each hidden state, while LSTM and RNN have a nonlinearity, which makes backpropagation through time a lot more stable for Mamba.
* Mamba can still be calculated in one forward pass through a parallel scan (prefix sum) operation, compared to e.g. RNNs and LSTMs where we need to calculate the previous timestep before we can calculate the next. The Mamba authors developed a hardware-aware algorithm in the same vein as FlashAttention which further improves efficiency.
* The authors mention that RNNs without time-wise nonlinearities such as QRNN are the most similar to Mamba, but those do not use state expansion or selective B and C params, and they use a heuristic gating mechanism, while the parameterizations and initializations of Mamba are based on principled SSM theory.

How it is different from Transformers?
Unlike transformers, where attention looks at every step of the sequence, the Mamba models have concept of memory at every step that tries to use whatever information is relevant from the past and foreget whatever is not for the current token.

A user commented that:

The model will have been trained to use the potentially gigabytes of retained state information to store and forward the relevant information as needed.

It’s just storing the history of the sequence in a far more efficient and highly structured way, and it will have been trained to very thoughtfully and efficiently retain recent information nearly perfectly and historical information if and when it may be relevant again or relates to recent information.

Think of it like this:

A transformer will need and work from the exact raw sequence data up to a history of N.

The transformer must then truncate and completely 100% forget anything beyond N tokens ago.

The state space model instead is efficiently encoding and storing the relevant information, and predicting what might still be relevant, so holding onto it.

Also, because things that are still relevant are inherently related in some way, that means it can really cleverly and efficiently compress and store that history in an efficient and structured way.

Another user commented:
After reading the Mamba paper, attention feels like a hack to avoid engineering a memory representation. “We don’t know how to make the network remember the content of the previous tokens so let’s just feed all of them into it over and over.” Hence the quadratic scaling with context size: each new token depends on all previous tokens instead of a fixed-size state.

Paper: https://arxiv.org/pdf/2312.00752.pdf

This blog has a great visual guide to Mamba (and even State Space Models): https://maartengrootendorst.substack.com?utm_source=navbar&utm_medium=web

## The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits

This paper from Microsoft introduces us to concept of training models using 1.58 bits, which drastically reduces the model size and increases the inference time. Why 1.58? Well, in the weights instead of using FP16 values like, 2.1745 etc, it uses ternary values [-1,0,1]. To represent these 3 values you need $log_2(3)$ bits, which is 1.58. 

So the quantisation happens while training the model itself and hence this is different from post training quantised models that have flooded the space of small sized models. Using ternary of [-1,0,1] also reduces the multiplication+addition operations in traditional linear layer with just addition operations (including subtraction for -1). So custom harware can take advantage of this to make inference even faster.

Checkout this video to see some details around the paper - https://www.youtube.com/watch?v=wCDGiys-nLA

The paper has comparisons on llama, for upto 3B parameters. But there are no open source models currently available in the open source community for testing. The future looks promising, specially for hosting these 1.58 Bit models on smaller GPUs or even CPUs, without the greate costs and low latencies. Seems perfect also for edge devices.

## Chameleon: Mixed-Modal Early-Fusion Foundation Models

A paper from meta AI that introduces `mixed-modal` models, which unlike the multi modal models handle the modalities differently. They are trained on interleaved, hence the term mix, where text and image can occur in any order. This is a more general purpose approach to handle text and image.

Only paper and no code or model released yet. Supposed to come in 2 sizes, 7B and 34B. Benchmarks reveal that it already surpasses llama 2 and is close to the mistral/mixtral models on text only tasks. For mix modality tasks they have created their own way of benchmarking mix of text and image generation. They compared to Gemini Pro and GPT-4V, after augmenting them for image generation (as they support multi modal input but only text output), by prompting them to generate image captions wherever image was required and then generating images with Dall-E 3. Chameleon was more preferred on human evaluation than Gemeni Pro + and GPT-4V, much larger models compared to it.

One interesting approach is the image tokenisation, which basically relies on a codebook (vector respresentation of image tokens) of 8192. So essentially 8192 tokens representing all possible images. This is along with 65,536 vocab for text data.

To do: Read more about image tokenisation using codebooks. Chameleon uses image tokeniser fomr this paper - Make-a-scene: Scene-based text-to-image generation with human priors