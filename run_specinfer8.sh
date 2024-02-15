# CUDA_VISIBLE_DEVICES=6 python test_specinfer.py --model  JackFram/llama-68m   --target meta-llama/Llama-2-7b-hf  --T 0.6 --P 1.0 --B 10  --DP 0.99 --W 32 --start 0 --end 200 --M 384 --growmap  /home/zhuoming/Sequoia2/growmaps/8x8-tree.pt --Mode greedy >> results_specinfer8A.log
#CUDA_VISIBLE_DEVICES=6 python test_specinfer.py --model  JackFram/llama-68m   --target meta-llama/Llama-2-7b-hf  --T 0.6 --P 1.0 --B 10  --DP 0.99 --W 32 --start 0 --end 200 --M 384 --growmap  /home/zhuoming/Sequoia2/growmaps/8x8-tree.pt --Mode greedy --dataset cnn >> results_specinfer8A.log
# CUDA_VISIBLE_DEVICES=6 python test_specinfer.py --model  JackFram/llama-68m   --target meta-llama/Llama-2-7b-hf  --T 0.6 --P 1.0 --B 10  --DP 0.99 --W 32 --start 0 --end 200 --M 384 --growmap  /home/zhuoming/Sequoia2/growmaps/8x8-tree.pt --Mode greedy --dataset openwebtext >> results_specinfer8A.log

# CUDA_VISIBLE_DEVICES=6 python test_specinfer.py --model  JackFram/llama-68m   --target meta-llama/Llama-2-13b-hf  --T 0.6 --P 1.0 --B 10  --DP 0.99 --W 32 --start 0 --end 200 --M 384 --growmap  /home/zhuoming/Sequoia2/growmaps/8x8-tree.pt --Mode greedy >> results_specinfer8A.log
#CUDA_VISIBLE_DEVICES=6 python test_specinfer.py --model  JackFram/llama-68m   --target meta-llama/Llama-2-13b-hf  --T 0.6 --P 1.0 --B 10  --DP 0.99 --W 32 --start 0 --end 200 --M 384 --growmap  /home/zhuoming/Sequoia2/growmaps/8x8-tree.pt --Mode greedy --dataset cnn >> results_specinfer8A.log
# CUDA_VISIBLE_DEVICES=6 python test_specinfer.py --model  JackFram/llama-68m   --target meta-llama/Llama-2-13b-hf  --T 0.6 --P 1.0 --B 10  --DP 0.99 --W 32 --start 0 --end 200 --M 384 --growmap  /home/zhuoming/Sequoia2/growmaps/8x8-tree.pt --Mode greedy --dataset openwebtext >> results_specinfer8A.log

# CUDA_VISIBLE_DEVICES=6 python test_specinfer.py --model  JackFram/llama-160m   --target meta-llama/Llama-2-13b-hf  --T 0.6 --P 1.0 --B 10  --DP 0.99 --W 32 --start 0 --end 200 --M 384 --growmap  /home/zhuoming/Sequoia2/growmaps/8x8-tree.pt --Mode greedy >> results_specinfer8A.log
#CUDA_VISIBLE_DEVICES=6 python test_specinfer.py --model  JackFram/llama-160m   --target meta-llama/Llama-2-13b-hf  --T 0.6 --P 1.0 --B 10  --DP 0.99 --W 32 --start 0 --end 200 --M 384 --growmap  /home/zhuoming/Sequoia2/growmaps/8x8-tree.pt --Mode greedy --dataset cnn >> results_specinfer8A.log
# CUDA_VISIBLE_DEVICES=6 python test_specinfer.py --model  JackFram/llama-160m   --target meta-llama/Llama-2-13b-hf  --T 0.6 --P 1.0 --B 10  --DP 0.99 --W 32 --start 0 --end 200 --M 384 --growmap  /home/zhuoming/Sequoia2/growmaps/8x8-tree.pt --Mode greedy --dataset openwebtext >> results_specinfer8A.log

CUDA_VISIBLE_DEVICES=6 python test_specinfer.py --model  princeton-nlp/Sheared-LLaMA-1.3B   --target lmsys/vicuna-33b-v1.3 --dataset cnn --T 0.6 --P 1.0 --B 10  --DP 0.99 --W 32 --start 0 --end 200 --M 384 --growmap  /home/zhuoming/Sequoia2/growmaps/8x8-tree.pt --Mode greedy >> results_specinfer8A.log
#CUDA_VISIBLE_DEVICES=6 python test_specinfer.py --model  JackFram/llama-160m   --target meta-llama/Llama-2-13b-hf  --T 0.6 --P 1.0 --B 10  --DP 0.99 --W 32 --start 0 --end 200 --M 384 --growmap  /home/zhuoming/Sequoia2/growmaps/8x8-tree.pt --Mode greedy --dataset cnn >> results_specinferA.log
#CUDA_VISIBLE_DEVICES=6 python test_specinfer.py --model  princeton-nlp/Sheared-LLaMA-1.3B   --target lmsys/vicuna-33b-v1.3  --T 0.6 --P 1.0 --B 10  --DP 0.99 --W 32 --start 0 --end 200 --M 384 --growmap  /home/zhuoming/Sequoia2/growmaps/8x8-tree.pt --Mode greedy --dataset openwebtext >> results_specinfer8A.log



