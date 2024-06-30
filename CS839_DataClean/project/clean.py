import os
import csv
import json
from matplotlib import pyplot as plt


def histogram(myDict):
    myDict = dict(sorted(myDict.items(), key=lambda item: item[1], reverse=True))
    categories = list(myDict.keys())
    values = list(myDict.values())

    plt.figure(figsize=(30, 15))
    plt.bar(categories, values, color='blue')

    plt.xlabel('Semantic Types')
    plt.ylabel('Number of Samples')

    plt.xticks(rotation=90)
    plt.subplots_adjust(bottom=0.2)  # 调整底部边缘空间


    plt.show()

folder_path = './sato_tables/multionly'

json_file = []
label_dict = {}

subfolder_list = os.listdir(folder_path)
for subfolder in subfolder_list:
    subfolder_path = os.path.join(folder_path, subfolder)
    # check folder type
    if not os.path.isdir(subfolder_path):
        continue

    print(f"Processing {subfolder_path}...")
    file_list = os.listdir(subfolder_path)
    for file in file_list:
        # get file path
        file_path = os.path.join(subfolder_path, file)
        # check file type
        if not os.path.isfile(file_path):
            continue

        with open(file_path, newline='') as csvfile:
            reader = csv.reader(csvfile)
            input_data = [row for row in reader]

        column_names = input_data[0]
        column_data = input_data[1:]
        for index, column_name in enumerate(column_names):
            if column_name not in label_dict:
                label_dict[column_name] = 1
            else:
                label_dict[column_name] += 1

            json_data = {
                "input": ",".join([row[index] for row in column_data]),
                "output": column_name
            }
            json_file.append(json_data)

# json_file_path = "data.json"
# with open(json_file_path, 'w') as file:
#     json.dump(json_file, file, indent=4)

# print(f"JSON file has been created at {json_file_path}")

histogram(label_dict)