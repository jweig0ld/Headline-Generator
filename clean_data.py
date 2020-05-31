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


