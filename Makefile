mlflow:
	uv run mlflow ui --backend-store-uri sqlite:///mlflow.db --host 0.0.0.0 --allowed-hosts "100.102.138.25:5000,100.102.138.25,localhost:*,127.0.0.1:*" --cors-allowed-origins "http://100.102.138.25:5000"

tokenizer:
	uv run python data/tokenizer_train.py

pretrain:
	CUDA_VISIBLE_DEVICES=0,1 uv run torchrun --nproc_per_node=2 finetuning/train.py --mode pretrain

train-sft:
	CUDA_VISIBLE_DEVICES=0,1 uv run torchrun --nproc_per_node=2 finetuning/train.py --mode sft

train-dpo:
	CUDA_VISIBLE_DEVICES=0,1 uv run torchrun --nproc_per_node=2 finetuning/train.py --mode dpo
