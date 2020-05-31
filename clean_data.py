import json


def json_to_text(json_file, destination_file):

    with open(json_file, 'r') as f:
        d = json.load(f)
        with open(destination_file, 'a') as g:
            for elem in d:
                g.write(str(elem['headline']) + "\n")
        g.close()
    f.close()

    print("Process completed.")


def generate_training_samples(n_samples, source_file, destination_file):

    with open(source_file, 'r') as f, open(destination_file, 'w') as g:
        lines = [line for line in f][:n_samples]
        g.writelines(lines)
    f.close()
    g.close()
    print("Process completed.")


generate_training_samples(1024, 'headlines.txt', 'headlines_1024.txt')