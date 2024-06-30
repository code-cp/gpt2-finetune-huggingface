from transformers import GPT2TokenizerFast, GPT2Config, GPT2LMHeadModel

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
config = GPT2Config.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("./models", config=config)

prompt = "Who are you\n"
print(f"{prompt}")
inputs = "Morty: Hi Rick." + prompt
inputs = tokenizer(inputs, return_tensors="pt")
outputs = model.generate(**inputs, do_sample=True, max_length=64, top_k=50, top_p=0.95, num_return_sequences=1)

result = tokenizer.decode(outputs[0])
lines = result.splitlines()  # Split the input string into lines
for line in lines:
    if line.strip().startswith("Rick:"):
        result = line.replace('Rick:', '')
        print(f"{result}")
        break 

