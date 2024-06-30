from transformers import GPT2TokenizerFast, GPT2Config, GPT2LMHeadModel

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
config = GPT2Config.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("./models", config=config)

inputs = tokenizer("Rick: I turned myself into a pickle, Morty!\nMorty: ", return_tensors="pt")
outputs = model.generate(**inputs, do_sample=True, max_length=64, top_k=50, top_p=0.95, num_return_sequences=4)

for idx in range(len(outputs)):
    result = tokenizer.decode(outputs[idx])
    print(f"sample {idx}: {result}")

