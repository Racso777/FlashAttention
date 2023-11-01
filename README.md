# FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness
This repository contains overview, explanation, and examples of FlashAttention as outlined in the official paper: https://arxiv.org/abs/2205.14135

Dao, T., Fu, D. Y., Ermon, S., Rudra, A., & Ré, C. (2022). FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness. arXiv preprint arXiv:2205.14135.

# Introduction
Transformer models have emerged as the most widely used architecture in applications such as natural language processing and image classification. Transformers have grown larger and deeper, but equipping them with longer context remains difficult, since the self-attention module at their heart
has time and memory complexity quadratic in sequence length. 

First question we should asked ourself is:
**"When we are using GPU to train our model, where does all the matrix stored and where are they trained at?"**

## Key Idea
### Compute-bound
the time taken by the operation is determined by how many arithmetic operations there are, while time accessing HBM is much smaller. Typical examples are matrix multiply with large inner
dimension, and convolution with large number of channels.
### Memory-bound
the time taken by the operation is determined by the number of memory accesses, while time spent in computation is much smaller. Examples include most other operations: elementwise (e.g., activation, dropout), and reduction (e.g., sum, softmax, batch norm, layer norm).

### I/O awareness
As an example, the A100 GPU has 40-80GB of high bandwidth memory (HBM) with bandwidth 1.5-2.0TB/s and 192KB of on-chip SRAM per each of 108 streaming multiprocessors with bandwidth estimated around 19TB/s [44, 45]. The on-chip
SRAM is an order of magnitude faster than HBM but many orders of magnitude smaller in size.

### Kernel fusion
The most common approach to accelerate memory-bound operations is kernel fusion: if there are multiple operations applied to the same input, the input can be loaded once from HBM, instead of
multiple times for each operation.

# Paper Overview
Our main goal is to avoid reading and writing the attention matrix to and from HBM.
This requires (i) computing the softmax reduction without access to the whole input (ii) not storing the large
intermediate attention matrix for the backward pass. We apply two well-established techniques to address
these challenges. (i) We restructure the attention computation to split the input into blocks and make several
passes over input blocks, thus incrementally performing the softmax reduction (also known as tiling). (ii) We
store the softmax normalization factor from the forward pass to quickly recompute attention on-chip in the
backward pass, which is faster than the standard approach of reading the intermediate attention matrix from
HBM.

![图片_20231031205400](https://github.com/Racso777/FlashAttention/assets/111296013/242ba47d-cf27-4b6e-b732-0f92f72d46df)


Even with the increased FLOPs due to recomputation,
our algorithm both runs faster (up to 7.6x on GPT-2 [67], Figure 1 right) and uses less memory—linear
in sequence length—than standard attention, thanks to the massively reduced amount of HBM access. 

## Difference between normal approach and FlashAttention Pseudocode
### Standard Attention Algorithm 

![图片_20231031205021](https://github.com/Racso777/FlashAttention/assets/111296013/72553fb4-43f7-421a-a54e-ae8f40857f45)
### Flash Attention Algorithm

![图片_20231031205015](https://github.com/Racso777/FlashAttention/assets/111296013/3a4e24df-f3fc-4dce-bad7-0e3036aea559)

### Difference
Standard attention implementations materialize the matrices S and P to HBM, which takes 𝑂(𝑁2) memory.


## Results
- Datasets:
 
- Model Architectures:

- Main Results:


**Discussion Question: Are there specific applications or domains where you think LoRA might be less suitable, and why?**

# Critical Analysis
**Advantages:**
- Drastically reduces number of trainable parameters compared to full fine-tuning. This enables fine-tuning huge models like GPT-3 with limited compute.


**Limitations:**
Compiling to CUDA. Our current approach to building IO-aware implementations of attention requires
writing a new CUDA kernel for each new attention implementation. This requires writing the attention
algorithm in a considerably lower-level language than PyTorch, and requires significant engineering effort.
Implementations may also not be transferrable across GPU architectures. These limitations suggest the
need for a method that supports writing attention algorithms in a high-level language (e.g., PyTorch), and
compiling to IO-aware implementations in CUDA—similar to efforts such as Halide in image processing [70].
IO-Aware Deep Learning. We believe that the IO-aware approach can extend beyond attention.
Attention is the most memory-intensive computation in Transformers, but every layer in a deep network
touches GPU HBM. We hope our work inspires IO-aware implementations of additional modules. We discuss
these potential extensions in Appendix D.
Multi-GPU IO-Aware Methods. Our IO-aware implementation of attention is optimal within constants for computing attention on a single GPU. However, the attention computation may be parallelizable
across multiple GPUs [72]. Using multiple GPUs adds an additional layer to IO analysis—accounting for
data transfer between GPUs. We hope our work inspires future work in this direction.


# Code Demonstration
Since I don't have a GPU to work on and this paper is specifcally targeted improvement on GPU memory fine-grained control, I don't have a notebook that could demonstrate this approach.
Please refer to https://github.com/Dao-AILab/flash-attention/tree/main this link for the official demonstration of FlashAttention.

# Reference
