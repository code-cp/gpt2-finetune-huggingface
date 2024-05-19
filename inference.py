from transformers import GPT2TokenizerFast, GPT2Config, GPT2LMHeadModel

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
config = GPT2Config.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("./models", config=config)

inputs = tokenizer("My name is Holmes, the address is ", return_tensors="pt")
outputs = model.generate(**inputs, do_sample=True, temperature=0.9, max_length=100)
result = tokenizer.decode(outputs[0])
print(result)

