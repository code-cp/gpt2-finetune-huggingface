import os 

def process_text(result_root, filename): 
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    lines1 = [line.strip().replace('MORTY: ', '') for line in lines if line.startswith('MORTY:')]
    lines2 = [line.strip().replace('Morty: ', '') for line in lines if line.startswith('Morty:')]

    data = lines1 + lines2
    data = [f"{line}\n" for line in data]

    result_path = filename.split("/")[-1]
    result_path = os.path.join(result_root, result_path)
    print(f"saving files to {result_path}")

    with open(result_path, 'w') as file:
        # Write the text to the file
        file.writelines(data)

    return result_path


