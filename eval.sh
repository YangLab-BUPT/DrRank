## Cross-Encoder

# CUDA_VISIBLE_DEVICES="0" python -u eval.py \
#   --input_path "data/DrRank_V2.jsonl" \
#   --model_name_or_path "BAAI/bge-reranker-v2-m3" \
#   --model_type "cross-encoder" \
#   --max_doctor_length 2048 \
#   --batch_size 8 \
# >> logs/DrRank_V2.bge-reranker-v2-m3.log

# CUDA_VISIBLE_DEVICES="0" python -u eval.py \
#   --input_path "data/DrRank_V2.jsonl" \
#   --model_name_or_path "BAAI/bge-reranker-v2-m3" \
#   --model_type "cross-encoder" \
#   --max_doctor_length 2048 \
#   --batch_size 96 \
#   --query_prefix "我是男性，我正在寻找" \
# >> logs/DrRank_V2.bge-reranker-v2-m3.log

# CUDA_VISIBLE_DEVICES="0" python -u eval.py \
#   --input_path "data/DrRank_V2.jsonl" \
#   --model_name_or_path "BAAI/bge-reranker-v2-m3" \
#   --model_type "cross-encoder" \
#   --max_doctor_length 2048 \
#   --batch_size 96 \
#   --query_prefix "我是女性，我正在寻找" \
# >> logs/DrRank_V2.bge-reranker-v2-m3.log

# CUDA_VISIBLE_DEVICES="0" python -u eval.py \
#   --input_path "data/DrRank_V2.jsonl" \
#   --model_name_or_path "BAAI/bge-reranker-v2-m3" \
#   --model_type "cross-encoder" \
#   --max_doctor_length 2048 \
#   --batch_size 96 \
#   --query_prefix "我来自城镇，我正在寻找" \
# >> logs/DrRank_V2.bge-reranker-v2-m3.log

# CUDA_VISIBLE_DEVICES="0" python -u eval.py \
#   --input_path "data/DrRank_V2.jsonl" \
#   --model_name_or_path "BAAI/bge-reranker-v2-m3" \
#   --model_type "cross-encoder" \
#   --max_doctor_length 2048 \
#   --batch_size 96 \
#   --query_prefix "我来自乡村，我正在寻找" \
# >> logs/DrRank_V2.bge-reranker-v2-m3.log


## LLM-based Logit Ranker

### Qwen2
#### RTX 4090

# CUDA_VISIBLE_DEVICES="0" python -u eval.py \
#   --input_path "data/DrRank_V2.jsonl" \
#   --model_name_or_path "Qwen/Qwen2-7B-Instruct" \
#   --model_type "llm-logit" \
#   --max_doctor_length 2048 \
#   --batch_size 5 \
#   --using_criteria \
# >> logs/DrRank_V2.Qwen2-7B-Instruct.log

### Qwen2.5
#### A800-SXM4-80G

# ( CUDA_VISIBLE_DEVICES="0" python -u eval.py \
#   --input_path "data/DrRank_V2.jsonl" \
#   --model_name_or_path "Qwen/Qwen2.5-1.5B-Instruct" \
#   --model_type "llm-logit" \
#   --max_doctor_length 2048 \
#   --batch_size 15 \
# >> logs/DrRank_V2.Qwen2.5-1.5B-Instruct.log &)

# ( CUDA_VISIBLE_DEVICES="1" python -u eval.py \
#   --input_path "data/DrRank_V2.jsonl" \
#   --model_name_or_path "Qwen/Qwen2.5-1.5B-Instruct" \
#   --model_type "llm-logit" \
#   --max_doctor_length 2048 \
#   --batch_size 10 \
#   --using_criteria \
# >> logs/DrRank_V2.Qwen2.5-1.5B-Instruct.criteria.log &)

# ( CUDA_VISIBLE_DEVICES="0" python -u eval.py \
#   --input_path "data/DrRank_V2.jsonl" \
#   --model_name_or_path "Qwen/Qwen2.5-7B-Instruct" \
#   --model_type "llm-logit" \
#   --max_doctor_length 2048 \
#   --batch_size 8 \
#   --label_nums 5 \
#   --using_criteria \
# >> logs/DrRank_V2.Qwen2.5-7B-Instruct.random_criteria.L5.log &)

# ( CUDA_VISIBLE_DEVICES="5" python -u eval.py \
#   --input_path "data/DrRank_V2.jsonl" \
#   --model_name_or_path "Qwen/Qwen2.5-7B-Instruct" \
#   --model_type "llm-logit" \
#   --max_doctor_length 2048 \
#   --batch_size 8 \
#   --label_nums 5 \
#   --query_prefix "我是男性，我正在寻找" \
#   --using_criteria \
# >> logs/DrRank_V2.Qwen2.5-7B-Instruct.criteria.L5.male.log &)

# ( CUDA_VISIBLE_DEVICES="6" python -u eval.py \
#   --input_path "data/DrRank_V2.jsonl" \
#   --model_name_or_path "Qwen/Qwen2.5-7B-Instruct" \
#   --model_type "llm-logit" \
#   --max_doctor_length 2048 \
#   --batch_size 8 \
#   --label_nums 5 \
#   --query_prefix "我是女性，我正在寻找" \
#   --using_criteria \
# >> logs/DrRank_V2.Qwen2.5-7B-Instruct.criteria.L5.female.log &)

# ( CUDA_VISIBLE_DEVICES="7" python -u eval.py \
#   --input_path "data/DrRank_V2.jsonl" \
#   --model_name_or_path "Qwen/Qwen2.5-7B-Instruct" \
#   --model_type "llm-logit" \
#   --max_doctor_length 2048 \
#   --batch_size 8 \
#   --label_nums 5 \
#   --query_prefix "我来自城镇，我正在寻找" \
#   --using_criteria \
# >> logs/DrRank_V2.Qwen2.5-7B-Instruct.criteria.L5.urban.log &)

# ( CUDA_VISIBLE_DEVICES="0" python -u eval.py \
#   --input_path "data/DrRank_V2.jsonl" \
#   --model_name_or_path "Qwen/Qwen2.5-7B-Instruct" \
#   --model_type "llm-logit" \
#   --max_doctor_length 2048 \
#   --batch_size 8 \
#   --label_nums 5 \
#   --query_prefix "我来自乡村，我正在寻找" \
#   --using_criteria \
# >> logs/DrRank_V2.Qwen2.5-7B-Instruct.criteria.L5.rural.log &)


# ( CUDA_VISIBLE_DEVICES="2,3" python -u eval.py \
#   --input_path "data/DrRank_V2.jsonl" \
#   --model_name_or_path "Qwen/Qwen2.5-14B-Instruct" \
#   --model_type "llm-logit" \
#   --max_doctor_length 2048 \
#   --batch_size 8 \
#   --label_nums 5 \
# >> logs/DrRank_V2.Qwen2.5-14B-Instruct.L5.log &)

# ( CUDA_VISIBLE_DEVICES="2,3" python -u eval.py \
#   --input_path "data/DrRank_V2.jsonl" \
#   --model_name_or_path "Qwen/Qwen2.5-14B-Instruct" \
#   --model_type "llm-logit" \
#   --max_doctor_length 2048 \
#   --batch_size 8 \
#   --using_criteria \
#   --label_nums 5 \
# >> logs/DrRank_V2.Qwen2.5-14B-Instruct.criteria.L5.log &)

# ( CUDA_VISIBLE_DEVICES="0,1" python -u eval.py \
#   --input_path "data/DrRank_V2.jsonl" \
#   --model_name_or_path "Qwen/Qwen2.5-14B-Instruct" \
#   --model_type "llm-logit" \
#   --max_doctor_length 2048 \
#   --batch_size 8 \
#   --using_criteria \
#   --label_nums 5 \
# >> logs/DrRank_V2.Qwen2.5-14B-Instruct.random_criteria.L5.log &)

# ( CUDA_VISIBLE_DEVICES="0,1,2,3" python -u eval.py \
#   --input_path "data/DrRank_V2.jsonl" \
#   --model_name_or_path "Qwen/Qwen2.5-32B-Instruct" \
#   --model_type "llm-logit" \
#   --max_doctor_length 2048 \
#   --batch_size 8 \
#   --label_nums 5 \
# >> logs/DrRank_V2.Qwen2.5-32B-Instruct.log &)

# ( CUDA_VISIBLE_DEVICES="4,5,6,7" python -u eval.py \
#   --input_path "data/DrRank_V2.jsonl" \
#   --model_name_or_path "Qwen/Qwen2.5-32B-Instruct" \
#   --model_type "llm-logit" \
#   --max_doctor_length 2048 \
#   --batch_size 8 \
#   --using_criteria \
#   --label_nums 5 \
# >> logs/DrRank_V2.Qwen2.5-32B-Instruct.criteria.log &)

# ( CUDA_VISIBLE_DEVICES="0,1,2,3" python -u eval.py \
#   --input_path "data/DrRank_V2.jsonl" \
#   --model_name_or_path "Qwen/Qwen2.5-32B-Instruct" \
#   --model_type "llm-logit" \
#   --max_doctor_length 2048 \
#   --batch_size 8 \
#   --using_criteria \
#   --label_nums 5 \
# >> logs/DrRank_V2.Qwen2.5-32B-Instruct.random_criteria.log &)