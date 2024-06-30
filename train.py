import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config, GPT2LMHeadModel, GPT2TokenizerFast
from transformers import TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
import matplotlib.pyplot as plt
from transformers import GPT2TokenizerFast, GPT2Config, GPT2LMHeadModel
import os 

# from utils import process_text 

def plot_training_metrics(trainer, metric_name):
    loss = []
    for l in trainer.state.log_history: 
        if metric_name in l: 
            loss.append(l[metric_name])
    plt.plot(loss, label=metric_name)
    plt.xlabel("Training Steps")
    plt.ylabel(metric_name.capitalize())
    plt.title(f"{metric_name.capitalize()} vs Training Steps")
    plt.legend()
    # plt.show()
    plt.savefig('./output/plot.png') 

def create_data_collator(tokenizer):
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False,
    )
    return data_collator

def train(): 
    # Load GPT-2 model and tokenizer
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    config = GPT2Config.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2", config=config)

    # Create dataset and data_collator
    results_root = os.path.join(os.getcwd(), "data")
    os.makedirs(results_root, exist_ok=True)

    results_path = "/home/nuc/workspace/build-gpt2/data/train.txt"
    train_dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=results_path,
        block_size=128,
    )

    results_path = "/home/nuc/workspace/build-gpt2/data/val.txt"
    val_dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=results_path,
        block_size=128,
    )

    data_collator = create_data_collator(tokenizer)

    # Configure Trainer instance
    training_args = TrainingArguments(
        output_dir="./output",
        overwrite_output_dir=True,
        num_train_epochs=100,
        per_device_train_batch_size=4,
        save_steps=10_000,
        save_total_limit=2,
        logging_steps=100,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    # Train the model
    train_result = trainer.train() 
    metrics = train_result.metrics
    trainer.save_model("./models")
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    metrics = train_result.metrics

    # Evaluate the model
    eval_results = trainer.evaluate()
    print(eval_results)

    # Visualize the model performance
    plot_training_metrics(trainer, "loss")

if __name__ == "__main__": 
    train()
