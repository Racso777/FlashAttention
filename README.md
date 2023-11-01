# FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness
This repository contains overview, explanation, and examples of FlashAttention as outlined in the official paper: https://arxiv.org/abs/2205.14135

Dao, T., Fu, D. Y., Ermon, S., Rudra, A., & Ré, C. (2022). FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness. arXiv preprint arXiv:2205.14135.

# Introduction
Transformer models have become the predominant architecture in fields like natural language processing and image classification, continually increasing in both size and depth. However, extending them to handle longer contextual information remains a challenge due to the quadratic time and memory complexity of the self-attention module, which is central to their structure.

The utilization of GPUs for machine learning training is a common practice, given their capability to handle parallel computations efficiently. This leads to an intriguing question: 

### In the context of GPU-accelerated training, where exactly are all the matrices stored, and where does the training of the model actually take place?

## Key Idea
### I/O in GPU
Here we take A100 GPU as an example, it has 40-80GB of high bandwidth memory (HBM) with bandwidth 1.5-2.0TB/s and 192KB of on-chip SRAM per each of 108 streaming multiprocessors with bandwidth estimated around 19TB/s. The SRAM located on-chip operates at a speed that is an order of magnitude higher than that of HBM, yet it is significantly smaller in size by several orders of magnitude.

![图片_20231031235612](https://github.com/Racso777/FlashAttention/assets/111296013/3fdf2ae5-2972-4c6a-aff2-972296828b89)

### Compute-bound
For these tasks, the duration is primarily influenced by the volume of arithmetic calculations, with the HBM access time being substantially lesser. Common instances encompass executing matrix multiplication when there is an extensive inner dimension and performing convolutions with a considerable quantity of channels.

### Memory-bound
Here, the duration of the task is predominantly dictated by the frequency of memory retrievals, and the time consumed by arithmetic calculations is relatively minor. This category includes a wide range of operations such as elementwise functions (activation functions, dropout operations) and aggregation functions (summation, softmax, batch normalization, layer normalization).

### Kernel fusion
A common method to optimize memory-intensive operations is through kernel fusion, which involves consolidating multiple computational procedures into a single kernel. When numerous operations are to be performed on the same set of data, this approach allows for a one-time load of the data from the High Bandwidth Memory (HBM), rather than requiring separate loads for each distinct operation. This leads to more efficient memory usage and reduced operational time.

# Paper Overview
The main goal of this paper is to avoid reading and writing the attention matrix to and from HBM.

Requirement:
1. Perform softmax reduction without full input access.
2. Avoid storing large intermediate attention matrix for the backward pass, using two established methods.
   
Solution:
1. Implement attention computation in blocks, using multiple passes for incremental softmax reduction (tiling).
2. Preserve the softmax normalization factor from the forward pass to expedite on-chip attention recalculation in the backward pass, avoiding the slower retrieval of the intermediate attention matrix from HBM.

![图片_20231031235608](https://github.com/Racso777/FlashAttention/assets/111296013/4da307c7-b482-4cb7-a1c7-73944b2be436)


Despite necessitating additional FLOPs for the purpose of recomputation, the algorithm not only executes more swiftly—achieving up to a 7.6x speedup on GPT-2—but also consumes less memory, with its usage scaling linearly with sequence length. This efficiency is attributed to the substantially diminished need for accessing High Bandwidth Memory (HBM).

## Difference between normal approach and FlashAttention Pseudocode
### Standard Attention Algorithm 

![图片_20231031205021](https://github.com/Racso777/FlashAttention/assets/111296013/72553fb4-43f7-421a-a54e-ae8f40857f45)
### Flash Attention Algorithm

![图片_20231031205015](https://github.com/Racso777/FlashAttention/assets/111296013/3a4e24df-f3fc-4dce-bad7-0e3036aea559)

![图片_20231101024314](https://github.com/Racso777/FlashAttention/assets/111296013/183ed5ce-dc3e-4982-8fce-03c8bce74ce7)


## Block-Sparse FlashAttention Algorithm

Given a predefined block sparsity mask M ∈ {0, 1}, we can easily adapt the flash attention algorithm to only compute the nonzero blocks of the attention matrix. The algorithm is identical, except we skip zero blocks. For large sequence lengths 𝑁, 𝑠 is often set to 𝑁^−1/2 or 𝑁^−1 log 𝑁, resultingin Θ(𝑁√𝑁) or Θ(𝑁 log 𝑁) IO complexity.

## Methods
This paper follows the MLPerf 1.1 guidelines to train BERT-large, utilizing the LAMB optimizer, a 3.75e-3 learning rate, a 448 batch size, and capping at 7100 steps. Training ceases when validation accuracy for masked language modeling hits 72.0%, with the run-time recorded. The training leverages FP16 precision with Apex AMP at O2 optimization.
The results are benchmarked against Nvidia’s reported training speed for MLPerf 1.1, ensuring a consistent train/validation data split and evaluating against the same 10,000 validation examples. The model is trained on eight A100-80GB GPUs, with run times ranging from 16 to 19 minutes over 10 runs.

For GPT-2, the paper utilizes standard implementations from Huggingface and Nvidia’s Megatron-LM, adhering to Megatron-LM’s training recipe. The model is trained on eight A100-40GB GPUs, with an effective batch size of 512, employing gradient accumulation to manage GPU memory constraints. Authors use AdamW optimizer, differing learning rates for GPT-2 small and medium, and a weight decay of 0.1, maintaining consistent hyperparameters across 400K steps and implementing mixed-precision training.
The dataset used is Openwebtext, processed with the GPT-2 BPE tokenizer. A random 0.5% of the dataset is set aside for validation, ensuring all models are evaluated on the same set. Training times for GPT-2 small range from 2.7 to 9.5 days, while GPT-2 medium takes between 6.9 to 21.0 days.

## Results
Training Speed: FlashAttention surpasses the MLPerf 1.1 speed record for BERT by 15%, triples GPT-2's speed compared to HuggingFace, and is 1.8 times faster than Megatron. It also accelerates the Long Range Arena (LRA) benchmark by 2.4 times.

![图片_20231101033530](https://github.com/Racso777/FlashAttention/assets/111296013/64442d71-3f47-4b99-b8ae-28b3923c9057)


Quality: FlashAttention enhances Transformers' capability to process longer sequences, improving their quality. It trains GPT-2 with a 4K context length quicker and more effectively than Megatron does with a 1K context length, achieving a 0.7 improvement in perplexity. Longer sequences yield a 6.4 point improvement in long-document classification tasks. FlashAttention also excels in challenging tasks like Path-X (16K sequence length) and block-sparse FlashAttention shows promise in tasks like Path-256 (64K sequence length).

![图片_20231101033533](https://github.com/Racso777/FlashAttention/assets/111296013/0c4d0677-7811-46d8-b9dd-8c2e668cd09a)


Benchmarking Attention: FlashAttention's memory footprint scales linearly with sequence length, performing up to three times faster than standard attention for sequences up to 2K. Block-sparse FlashAttention’s runtime also scales linearly and outperforms all existing approximate attention methods.

![图片_20231101033537](https://github.com/Racso777/FlashAttention/assets/111296013/f5696613-312e-4519-b68c-a90bcfc5fc1a)

### Discussion Question: FlashAttention appears to be a versatile and valuable tool, especially when implemented on GPU-supported models, which encompasses a majority of current models. What potential drawbacks might be associated with this model?

**Limitations:**
CUDA Compilation: We need a new CUDA kernel for each variant of attention, requiring low-level programming and extensive engineering, which may not be consistent across GPU architectures. A high-level language for writing attention algorithms, translatable to IO-aware CUDA implementations, is needed.

Multi-GPU IO-Aware Methods: While our attention implementation is nearly optimal for single-GPU use, extending and optimizing it for multi-GPU environments, including accounting for inter-GPU data transfers, represents an exciting area for future research.

# Code Demonstration
If we want to train a model using this approach, we could clone the repo and run the python file: https://github.com/Dao-AILab/flash-attention/tree/main/training/run.py

The test dataset that we could use and train is in this file: https://github.com/Dao-AILab/flash-attention/blob/main/training/tests/datamodules/test_language_modeling_hf.py

Please refer to https://github.com/Dao-AILab/flash-attention/tree/main for the official demonstration and the source code of FlashAttention.

# More information on FlashAttention
Flashier Attention blog: https://www.adept.ai/blog/flashier-attention 

Tri Dao’s talk: https: //www.youtube.com/watch?v=gMOAud7hZg4

Tri Dao’s talk: https: //www.youtube.com/watch?v=FThvfkXWqtE

ELI5: FlashAttention: https://gordicaleksa.medium.com/eli5-flash-attention-5c44017022ad

Huggingface: https://huggingface.co/docs/text-generation-inference/conceptual/flash_attention

# Reference
Dao, T., Fu, D. Y., Ermon, S., Rudra, A., & Ré, C. (2022). FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness. arXiv preprint arXiv:2205.14135.

Gordić, A. (2022, July 18). ELI5: FlashAttention. Medium. https://gordicaleksa.medium.com/eli5-flash-attention-5c44017022ad

Hugging Face. (2023). Flash Attention. Hugging Face Documentation. https://huggingface.co/docs/text-generation-inference/conceptual/flash_attention

Tri Dao. (2023, January 1). FlashAttention - Tri Dao | Stanford MLSys #67. YouTube. https://www.youtube.com/watch?v=gMOAud7hZg4

