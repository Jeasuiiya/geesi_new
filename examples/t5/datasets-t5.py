from datasets import Dataset, load_dataset, load_from_disk
dataset = load_dataset("oscar", "unshuffled_deduplicated_no", split="train",trust_remote_code=True)
dataset.save_to_disk("./norwegian-t5-base") # 保存到该目录下
dataset