CUDA_VISIBLE_DEVICES=5 python test_greedyS.py --model  JackFram/llama-68m   --target meta-llama/Llama-2-7b-hf  --T 1.0 --P 0.9 --B 10  --DP 0.99 --W 32 --start 0 --end 200 --M 384 --growmap /home/zhuoming/Sequoia2/68m_7b/growmaps/A100-CNN-68m-7b-stochastic.pt  --Mode greedy --dataset cnn >> resultsZ.log

CUDA_VISIBLE_DEVICES=5 python test_greedyS.py --model  JackFram/llama-68m   --target meta-llama/Llama-2-7b-hf  --T 1.0 --P 0.8 --B 10  --DP 0.99 --W 32 --start 0 --end 200 --M 384 --growmap /home/zhuoming/Sequoia2/68m_7b/growmaps/A100-CNN-68m-7b-stochastic.pt  --Mode greedy --dataset cnn >> resultsZ.log

# CUDA_VISIBLE_DEVICES=5 python test_greedyS.py --model  JackFram/llama-68m   --target meta-llama/Llama-2-7b-hf  --T 0.6 --P 0.8 --B 10  --DP 0.99 --W 32 --start 0 --end 200 --M 384 --growmap /home/zhuoming/Sequoia2/68m_7b/growmaps/A100-CNN-68m-7b-stochastic.pt  --Mode greedy --dataset cnn >> resultsZ.log

# CUDA_VISIBLE_DEVICES=5 python test_greedyS.py --model  JackFram/llama-68m   --target meta-llama/Llama-2-7b-hf  --T 0.6 --P 0.9 --B 10  --DP 0.99 --W 32 --start 0 --end 200 --M 384 --growmap /home/zhuoming/Sequoia2/68m_7b/growmaps/A100-CNN-68m-7b-stochastic.pt  --Mode greedy --dataset cnn >> resultsZ.log

# CUDA_VISIBLE_DEVICES=5 python test_greedyS.py --model  JackFram/llama-68m   --target meta-llama/Llama-2-7b-hf  --T 1.0 --P 1.0 --B 10  --DP 0.99 --W 32 --start 0 --end 200 --M 384 --growmap /home/zhuoming/Sequoia2/68m_7b/growmaps/A100-CNN-68m-7b-stochastic.pt  --Mode greedy --dataset cnn >> resultsZ.log

# CUDA_VISIBLE_DEVICES=5 python test_greedyS.py --model  JackFram/llama-68m   --target meta-llama/Llama-2-7b-hf  --T 0.8 --P 1.0 --B 10  --DP 0.99 --W 32 --start 0 --end 200 --M 384 --growmap /home/zhuoming/Sequoia2/68m_7b/growmaps/A100-CNN-68m-7b-stochastic.pt  --Mode greedy --dataset cnn >> resultsZ.log

# CUDA_VISIBLE_DEVICES=5 python test_greedyS.py --model  JackFram/llama-68m   --target meta-llama/Llama-2-7b-hf  --T 0.6 --P 1.0 --B 10  --DP 0.99 --W 32 --start 0 --end 200 --M 384 --growmap /home/zhuoming/Sequoia2/68m_7b/growmaps/A100-CNN-68m-7b-stochastic.pt  --Mode greedy --dataset cnn >> resultsZ.log

# CUDA_VISIBLE_DEVICES=5 python test_greedyS.py --model  JackFram/llama-68m   --target meta-llama/Llama-2-7b-hf  --T 0.4 --P 1.0 --B 10  --DP 0.99 --W 32 --start 0 --end 200 --M 384 --growmap /home/zhuoming/Sequoia2/68m_7b/growmaps/A100-CNN-68m-7b-stochastic.pt  --Mode greedy --dataset cnn >> resultsZ.log

# CUDA_VISIBLE_DEVICES=5 python test_greedyS.py --model  JackFram/llama-68m   --target meta-llama/Llama-2-7b-hf  --T 0.2 --P 1.0 --B 10  --DP 0.99 --W 32 --start 0 --end 200 --M 384 --growmap /home/zhuoming/Sequoia2/68m_7b/growmaps/A100-CNN-68m-7b-stochastic.pt  --Mode greedy --dataset cnn >> resultsZ.log