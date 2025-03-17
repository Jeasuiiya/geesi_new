from transformers import T5Config

config = T5Config.from_pretrained("google/t5-v1_1-base", vocab_size=32_000)
config.save_pretrained("./norwegian-t5-base")