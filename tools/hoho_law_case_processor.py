import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# INPUT_DIR = "H:/AI/data/法律/LeCaRD/documents/documents"
# OUTPUT_DIR = "../outputs/data"
# RANDOM_STATE = 6666
# max_input_lengths = []
# max_target_lengths = []
# output_contents = []

# # num = 0

# for item in os.listdir(INPUT_DIR):
#     item_path = os.path.join(INPUT_DIR, item)
#     if os.path.isdir(item_path):
#         for file in os.listdir(item_path):
#             file_path = os.path.join(item_path, file)

#             with open(file_path, "r", encoding = "utf8") as f:
#                 content = f.read()
#                 json_obj = json.loads(content)
#                 case_description = json_obj.get("ajjbqk", "")
#                 judgement = json_obj.get("pjjg", "")
#                 if len(case_description) > 0 and len(judgement) > 0:
#                     case_info = {"ajjbqk": case_description, "pjjg": judgement}
#                     case_str = json.dumps(case_info, ensure_ascii = False)
#                     output_contents.append(case_str)
                    
#                     max_input_lengths.append(len(case_description))
#                     max_target_lengths.append(len(judgement))

#                     # num += 1

#     # if num > 100:
#     #     break


# print(f"max_input_length: {np.max(max_input_lengths)}")
# print(f"max_target_length: {np.max(max_target_lengths)}")


# train_law_cases, val_law_cases = train_test_split(output_contents, test_size = 0.2, random_state = RANDOM_STATE)


# train_json_str = "\n".join(train_law_cases)
# val_json_str = "\n".join(val_law_cases)

# with open(OUTPUT_DIR + "/train_law_cases.json", "w", encoding = "utf8") as f1:
#     f1.write(train_json_str)

# with open(OUTPUT_DIR + "/val_law_cases.json", "w", encoding = "utf8") as f2:
#     f2.write(val_json_str)


# fig, axes = plt.subplots(figsize = (12, 6), ncols = 1, nrows = 2)
# ax1 = axes[0]
# ax2 = axes[1]
# ax1.set_title("input length")
# ax1.set_xlabel("length")
# ax1.set_ylabel("count")
# ax1.hist(max_input_lengths, bins = 1000)
# ax2.set_title("target length")
# ax2.set_xlabel("length")
# ax2.set_ylabel("count")
# ax2.hist(max_target_lengths, bins = 1000)
# plt.savefig(OUTPUT_DIR + "/law_cases_length_distribution.png")

# 根据内容长度筛选数据
file_path = "../outputs/data/train_law_cases.json"
result_list = []
with open(file_path, "r", encoding = "utf8") as f:
    content = f.read()
    law_cases = content.split("\n")
    for case_content in law_cases:
        case_info = json.loads(case_content)
        case_description = case_info.get("ajjbqk", "")
        judgement = case_info.get("pjjg", "")
        if len(case_description) < 300 and len(judgement) < 300:
            result_list.append(case_content)

result_str = "\n".join(result_list)
with open("../outputs/data/train_law_cases_len300.json", "w", encoding = "utf8") as f:
    f.write(result_str)