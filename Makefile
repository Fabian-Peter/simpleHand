train:
	torchrun --nproc_per_node=1 train.py 

eval:
	python3 infer_to_json.py epoch_200
