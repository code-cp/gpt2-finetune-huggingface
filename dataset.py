from datasets import load_dataset
import re
import os 

def process_raw_data(data_dir):
    data = []
    # Load the dataset
    ds = load_dataset("Prarabdha/Rick_and_Morty_Transcript")
    for row in ds['train']:
        speaker = row['speaker']
        if speaker is None or speaker == "????": 
            speaker = "???" 
        if ":" not in (speaker + row['dialouge']):
            speaker += ": "
        line = speaker + row['dialouge'] 
        line = line.replace('\n', '')
        line = re.sub(r'\s+', ' ', line)
        line += "\n"
        data.append(line)

    ratio = 0.9 
    total_len = len(data)
    split_idx = int(total_len * ratio)
    train_data = data[:split_idx]
    val_data = data[split_idx:]

    filename = os.path.join(data_dir, "train.txt")
    with open(filename, "w") as file: 
        file.writelines(train_data) 

    filename = os.path.join(data_dir, "val.txt")
    with open(filename, "w") as file: 
        file.writelines(val_data) 

    return 

if __name__ == "__main__": 
    data_dir = "./data"
    process_raw_data(data_dir)