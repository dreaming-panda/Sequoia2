CUDA_VISIBLE_DEVICES=8 python benchmark_inference.py --P 192 --D 256 --model meta-llama/Llama-2-7b-hf

CUDA_VISIBLE_DEVICES=8 python benchmark_inference.py --P 192 --D 512 --model meta-llama/Llama-2-7b-hf

CUDA_VISIBLE_DEVICES=8 python benchmark_inference.py --P 192 --D 256 --model meta-llama/Llama-2-13b-hf

CUDA_VISIBLE_DEVICES=8 python benchmark_inference.py --P 192 --D 512 --model meta-llama/Llama-2-13b-hf






