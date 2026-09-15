import os
import time
import argparse
from random import randint, seed
from nanovllm import LLM, SamplingParams
# from vllm import LLM, SamplingParams


def main():
    parser = argparse.ArgumentParser(description="Throughput benchmark for nano-vllm.")
    parser.add_argument('--model', type=str, default=os.path.expanduser("~/huggingface/Qwen3-0.6B/"))
    parser.add_argument('--enforce-eager', action='store_true', default=False, help="Enforce eager execution mode.")
    parser.add_argument('--use-triton', action='store_true', default=False, help="Use fused Triton kernels for Add+RMSNorm and SiluAndMul.")
    args = parser.parse_args()

    seed(0)
    num_seqs = 256
    max_input_len = 1024
    max_ouput_len = 1024

    path = args.model
    llm = LLM(path, enforce_eager=args.enforce_eager, max_model_len=4096, use_triton=args.use_triton)

    prompt_token_ids = [[randint(0, 10000) for _ in range(randint(100, max_input_len))] for _ in range(num_seqs)]
    sampling_params = [SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=randint(100, max_ouput_len)) for _ in range(num_seqs)]
    # uncomment the following line for vllm
    # prompt_token_ids = [dict(prompt_token_ids=p) for p in prompt_token_ids]

    llm.generate(["Benchmark: "], SamplingParams())
    t = time.time()
    llm.generate(prompt_token_ids, sampling_params, use_tqdm=False)
    t = (time.time() - t)
    total_tokens = sum(sp.max_tokens for sp in sampling_params)
    throughput = total_tokens / t
    print(f"Total: {total_tokens}tok, Time: {t:.2f}s, Throughput: {throughput:.2f}tok/s")


if __name__ == "__main__":
    main()
