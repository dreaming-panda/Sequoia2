#CUDA_VISIBLE_DEVICES=0 python testbed.py --model  JackFram/llama-68m   --target meta-llama/Llama-2-7b-hf  --T 0.6 --P 1.0 --B 10  --DP 0.99 --W 32 --start 0 --end 200 --M 384 --growmap /home/zhuoming/Sequoia2/68m_7b/growmaps/A100-C4-68m-7b-stochastic.pt  --Mode greedy >> resultsv.log
CUDA_VISIBLE_DEVICES=0 python testbed.py --model  JackFram/llama-68m   --target meta-llama/Llama-2-7b-hf  --T 0.6 --P 1.0 --B 10  --DP 0.99 --W 32 --start 0 --end 200 --M 384 --growmap ./68m_7b/growmaps/A100-CNN-68m-7b-stochastic-32.pt  --Mode greedy --dataset cnn
CUDA_VISIBLE_DEVICES=0 python testbed.py --model  JackFram/llama-68m   --target meta-llama/Llama-2-7b-hf  --T 0.6 --P 1.0 --B 10  --DP 0.99 --W 32 --start 0 --end 200 --M 384 --growmap ./68m_7b/growmaps/A100-CNN-68m-7b-stochastic-64.pt  --Mode greedy --dataset cnn
# CUDA_VISIBLE_DEVICES=0 python testbed.py --model  JackFram/llama-68m   --target meta-llama/Llama-2-7b-hf  --T 0.6 --P 1.0 --B 10  --DP 0.99 --W 32 --start 0 --end 200 --M 384 --growmap /home/zhuoming/Sequoia2/68m_7b/growmaps/A100-OpenWebText-68m-7b-stochastic.pt  --Mode greedy --dataset openwebtext >> resultsv.log
# #CUDA_VISIBLE_DEVICES=0 python testbed.py --model  JackFram/llama-68m   --target meta-llama/Llama-2-13b-hf  --T 0.6 --P 1.0 --B 10  --DP 0.99 --W 32 --start 0 --end 200 --M 384 --growmap /home/zhuoming/Sequoia2/68m_13b/growmaps/A100-C4-68m-13b-stochastic.pt  --Mode baseline >> resultsv.log
# CUDA_VISIBLE_DEVICES=0 python testbed_greedy.py --model  JackFram/llama-68m   --target meta-llama/Llama-2-7b-hf  --T 0.6 --P 1.0 --B 10  --DP 0.99 --W 32 --start 0 --end 200 --M 384 --growmap /home/zhuoming/Sequoia2/68m_7b/growmaps/A100-C4-68m-7b-greedy.pt  --Mode greedy >> resultsv.log                 
# CUDA_VISIBLE_DEVICES=0 python testbed_greedy.py --model  JackFram/llama-68m   --target meta-llama/Llama-2-7b-hf  --T 0.6 --P 1.0 --B 10  --DP 0.99 --W 32 --start 0 --end 200 --M 384 --growmap /home/zhuoming/Sequoia2/68m_7b/growmaps/A100-CNN-68m-7b-greedy.pt  --Mode greedy --dataset cnn >> resultsv.log
# CUDA_VISIBLE_DEVICES=0 python testbed_greedy.py --model  JackFram/llama-68m   --target meta-llama/Llama-2-7b-hf  --T 0.6 --P 1.0 --B 10  --DP 0.99 --W 32 --start 0 --end 200 --M 384 --growmap /home/zhuoming/Sequoia2/68m_7b/growmaps/A100-OpenWebText-68m-7b-greedy.pt  --Mode greedy --dataset openwebtext >> resultsv.log
# #CUDA_VISIBLE_DEVICES=0 python testbed_greedy.py --model  JackFram/llama-68m   --target meta-llama/Llama-2-13b-hf  --T 0.6 --P 1.0 --B 10  --DP 0.99 --W 32 --start 0 --end 200 --M 384 --growmap /home/zhuoming/Sequoia2/68m_13b/growmaps/A100-OpenWebText-68m-13b-greedy.pt  --Mode baseline >> resultsv.log

# CUDA_VISIBLE_DEVICES=0 python testbed.py --model  JackFram/llama-68m   --target meta-llama/Llama-2-13b-hf  --T 0.6 --P 1.0 --B 10  --DP 0.99 --W 32 --start 0 --end 200 --M 384 --growmap /home/zhuoming/Sequoia2/68m_13b/growmaps/A100-C4-68m-13b-stochastic.pt  --Mode greedy >> resultsv.log
# CUDA_VISIBLE_DEVICES=0 python testbed.py --model  JackFram/llama-68m   --target meta-llama/Llama-2-13b-hf  --T 0.6 --P 1.0 --B 10  --DP 0.99 --W 32 --start 0 --end 200 --M 384 --growmap /home/zhuoming/Sequoia2/68m_13b/growmaps/A100-CNN-68m-13b-stochastic.pt  --Mode greedy --dataset cnn >> resultsv.log
# CUDA_VISIBLE_DEVICES=0 python testbed.py --model  JackFram/llama-68m   --target meta-llama/Llama-2-13b-hf  --T 0.6 --P 1.0 --B 10  --DP 0.99 --W 32 --start 0 --end 200 --M 384 --growmap /home/zhuoming/Sequoia2/68m_13b/growmaps/A100-OpenWebText-68m-13b-stochastic.pt  --Mode greedy --dataset openwebtext >> resultsv.log
# #CUDA_VISIBLE_DEVICES=0 python testbed.py --model  JackFram/llama-68m   --target meta-llama/Llama-2-13b-hf  --T 0.6 --P 1.0 --B 10  --DP 0.99 --W 32 --start 0 --end 200 --M 384 --growmap /home/zhuoming/Sequoia2/68m_13b/growmaps/A100-C4-68m-13b-stochastic.pt  --Mode baseline >> resultsv.log
# CUDA_VISIBLE_DEVICES=0 python testbed_greedy.py --model  JackFram/llama-68m   --target meta-llama/Llama-2-13b-hf  --T 0.6 --P 1.0 --B 10  --DP 0.99 --W 32 --start 0 --end 200 --M 384 --growmap /home/zhuoming/Sequoia2/68m_13b/growmaps/A100-C4-68m-13b-greedy.pt  --Mode greedy >> resultsv.log                 
# CUDA_VISIBLE_DEVICES=0 python testbed_greedy.py --model  JackFram/llama-68m   --target meta-llama/Llama-2-13b-hf  --T 0.6 --P 1.0 --B 10  --DP 0.99 --W 32 --start 0 --end 200 --M 384 --growmap /home/zhuoming/Sequoia2/68m_13b/growmaps/A100-CNN-68m-13b-greedy.pt  --Mode greedy --dataset cnn >> resultsv.log
# CUDA_VISIBLE_DEVICES=0 python testbed_greedy.py --model  JackFram/llama-68m   --target meta-llama/Llama-2-13b-hf  --T 0.6 --P 1.0 --B 10  --DP 0.99 --W 32 --start 0 --end 200 --M 384 --growmap /home/zhuoming/Sequoia2/68m_13b/growmaps/A100-OpenWebText-68m-13b-greedy.pt  --Mode greedy --dataset openwebtext >> resultsv.log
# #CUDA_VISIBLE_DEVICES=0 python testbed_greedy.py --model  JackFram/llama-68m   --target meta-llama/Llama-2-13b-hf  --T 0.6 --P 1.0 --B 10  --DP 0.99 --W 32 --start 0 --end 200 --M 384 --growmap /home/zhuoming/Sequoia2/68m_13b/growmaps/A100-OpenWebText-68m-13b-greedy.pt  --Mode baseline >> resultsv.log


# CUDA_VISIBLE_DEVICES=0 python testbed.py --model  JackFram/llama-160m   --target meta-llama/Llama-2-13b-hf  --T 0.6 --P 1.0 --B 10  --DP 0.99 --W 32 --start 0 --end 200 --M 384 --growmap /home/zhuoming/Sequoia2/160m_13b/growmaps/A100-C4-160m-13b-stochastic.pt  --Mode greedy >> resultsv.log
# CUDA_VISIBLE_DEVICES=0 python testbed.py --model  JackFram/llama-160m   --target meta-llama/Llama-2-13b-hf  --T 0.6 --P 1.0 --B 10  --DP 0.99 --W 32 --start 0 --end 200 --M 384 --growmap /home/zhuoming/Sequoia2/160m_13b/growmaps/A100-CNN-160m-13b-stochastic.pt  --Mode greedy --dataset cnn >> resultsv.log
# CUDA_VISIBLE_DEVICES=0 python testbed.py --model  JackFram/llama-160m   --target meta-llama/Llama-2-13b-hf  --T 0.6 --P 1.0 --B 10  --DP 0.99 --W 32 --start 0 --end 200 --M 384 --growmap /home/zhuoming/Sequoia2/160m_13b/growmaps/A100-OpenWebText-160m-13b-stochastic.pt  --Mode greedy --dataset openwebtext >> resultsv.log
# #CUDA_VISIBLE_DEVICES=0 python testbed.py --model  JackFram/llama-160m   --target meta-llama/Llama-2-13b-hf  --T 0.6 --P 1.0 --B 10  --DP 0.99 --W 32 --start 0 --end 200 --M 384 --growmap /home/zhuoming/Sequoia2/160m_13b/growmaps/A100-C4-160m-13b-stochastic.pt  --Mode baseline
# CUDA_VISIBLE_DEVICES=0 python testbed_greedy.py --model  JackFram/llama-160m   --target meta-llama/Llama-2-13b-hf  --T 0.6 --P 1.0 --B 10  --DP 0.99 --W 32 --start 0 --end 200 --M 384 --growmap /home/zhuoming/Sequoia2/160m_13b/growmaps/A100-C4-160m-13b-greedy.pt  --Mode greedy >> resultsv.log
# CUDA_VISIBLE_DEVICES=0 python testbed_greedy.py --model  JackFram/llama-160m   --target meta-llama/Llama-2-13b-hf  --T 0.6 --P 1.0 --B 10  --DP 0.99 --W 32 --start 0 --end 200 --M 384 --growmap /home/zhuoming/Sequoia2/160m_13b/growmaps/A100-CNN-160m-13b-greedy.pt  --Mode greedy --dataset cnn >> resultsv.log
# CUDA_VISIBLE_DEVICES=0 python testbed_greedy.py --model  JackFram/llama-160m   --target meta-llama/Llama-2-13b-hf  --T 0.6 --P 1.0 --B 10  --DP 0.99 --W 32 --start 0 --end 200 --M 384 --growmap /home/zhuoming/Sequoia2/160m_13b/growmaps/A100-OpenWebText-160m-13b-greedy.pt  --Mode greedy --dataset openwebtext >> resultsv.log
# #CUDA_VISIBLE_DEVICES=0 python testbed_greedy.py --model  JackFram/llama-160m   --target meta-llama/Llama-2-13b-hf  --T 0.6 --P 1.0 --B 10  --DP 0.99 --W 32 --start 0 --end 200 --M 384 --growmap /home/zhuoming/Sequoia2/160m_13b/growmaps/A100-OpenWebText-160m-13b-greedy.pt  --Mode baseline


