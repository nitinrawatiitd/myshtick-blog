---
title: "Good intuitions about LLMs"
date: 2023-09-23
author: Nitin
tags: [llm]
---

These few intuitions about LLMs at scale and training are helpful

Note: These notes are from presentation by Hyung Won Chung, Research Scientist at OpenAI.
* talk: https://youtu.be/dbo3kNKPaUA?feature=shared
* slides: https://docs.google.com/presentation/d/1636wKStYdT_yRPbJNrf8MLKpQghuWGDmyHinHhAKeXY/edit#slide=id.g2885e521b53_0_0

## Some abilities emerge with scale

It has been observed that some capabilities only appear after a certain point in scale. 
![](/img/emergent_abilities_llms.png "Emergent abilities in large language models (Jason Wei, Yi Tay, Rishi Bommasani, Colin Raffel, Barret Zoph et al. (2022))")

In gen ai space, it could appear that an idea doesn't work, but that could be just it doesn't work "yet". Need to constant learn and unlearn. Many ideas get outdated and invalidated at larger scale. We need to constantly unlearn intuitions built on such invalidated ideas. With less to unlearn, newcomers can have advantages over more experienced ones. This is an interesting neutralizing force.

So, to stay ahead of the scaling curve:
* Document experiments that failed because of insufficient “intelligence”
* Do not declare failure yet and make it easy to rerun in the future
* As soon as the new model comes out, rerun them
* Learn what works and what doesn’t
* Update your intuition on emergent abilities and scale

## How is the scaling actually done?

All LLMs so far use Transformer architecture. 

From first-principles view:
* Scaling Transformer means efficiently doing matmuls with many machines
* This involves distributing all the matrices (or arrays) involved in the Transformer layer to various machines
* Do so while minimizing the communication between machines

Matrix multiplication mapped to multiple machines
![](/img/matmul_multi.png "Matrix multiplication on multiple machines")

Let's focus on what machine does
![](/img/matmul_one.png "Matrix multiplication on multiple machines")

The code for the attention layers, where most of the matmul happens, needs to be written in a way so as to take advantage of parallel matmuls. That hardware to axis mapping defines parallelism.

Iteration on pre-training is very expensive and scaling to the largest scale ever is very, very hard. Problems can happen and every hour you don't make a decision, you are making expensive hardware lie idle.

## Training the model
Scaling doesn’t solve all problems. We also need post-training

We can’t talk to the pretrained model directly. Pre-trained models always generate something that is a natural continuation of the prompts even if the prompts are malicious

![](/img/pretrained_vs_instructiontuned.png "Pretrained vs instructiontuned")

So to address the limitations of a simple pretrained model, there are some post training steps that you can do -
![](/img/post_training_steps.png "Post training steps")

### Instruction fine tuning

Frame all tasks in the form of (natural language instruction) -> (natural language response) mapping. The great thing about this formalisation is that it can be used for any kind of task, unrelated even. This symbolises the true benefit of LLMs - they are not built for one task like in traditional machine learning, they can do multiple things.

So for an unseen task, the model just needs to respond to the natural language instruction.
![](/img/unified_tasks_instruction_tuning.png "Unified tasks in instruction tuning")

Scaling the number of tasks and model size improves the performance. 
![](/img/intructiontuning_scaling.png "Scaling of intruction tuning")

Instruction fine-tuning is highly effective but it has inherent limitations. That is where the next step come in

### RLHF
Suppose the input to an LLM is:
Write a letter to a 5-year-old boy from Santa Clause explaining that Santa is not real. Convey gently so as not to break his heart

What should the output be? There is no single correct answer.

So the observations are:
* Increasingly we want to teach models more abstract behaviors
* Objective function of instruction finetuning seems to be the “bottleneck” of teaching these behaviors
* The maximum likelihood objective is “predefined” function (i.e. no learnable parameter)
* Can we parameterize the objective function and learn it?

In RL, we try to maximize the expected reward function. Reward is the objective function. We can learn the reward: reward model.We know how to do supervised learning with neural network well. Let’s use neural net to represent the reward model.

Reward Model (RM) training data: which completion is better? Humans label which completion is preferred. This setup aims to align models to the human preference.
![](/img/rm_data.png "Data for reward model")

Once we have a reward model, we can use it in RL to learn the language model parameters that maximizes the expected reward. We can use policy gradient algorithms such as PPO to compute the gradients and update the model parameters.

### Why should we keep studying RLHF?

RLHF is natural next step in the evolution of machine learning models. If something is so principled, we should keep at it until it works.

This diagram helps capture that evolution. Super useful to understand where supervised fine tuned model (GPT-3) ends in supervised learning and where the RLHF comes into figure (InstructGPT and newer versions of GPT)
![](/img/ml_progress.png "Progress of machine learning")



