import os
import platform
import signal
import time

from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

import torch
import torch.multiprocessing as mp

MODEL_PATH = "/root/autodl-tmp/models/chatglm-6b/"
DATA_PATH = "/root/autodl-tmp/data/txt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EXCEPTION_FILES = ["1025295.wav.txt"]

PROMPT_TEMPLATE = """
{customer_say}\n\n请对上文对话进行主题总结，要求简洁精要，限制10字以内。
"""

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code = True)
model = AutoModel.from_pretrained(MODEL_PATH, trust_remote_code = True).half().cuda()
model = model.eval()


# print(f"model: {id(model)}, name: {__name__}, pid: {os.getpid()}")


print('成功加载GLM模型')
os_name = platform.system()
clear_command = 'cls' if os_name == 'Windows' else 'clear'
stop_stream = False
#import docx
import pandas as pd


def torch_gc(DEVICE):
    if torch.cuda.is_available():
        with torch.cuda.device(DEVICE):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    elif torch.backends.mps.is_available():
        try:
            from torch.mps import empty_cache
            empty_cache()
        except Exception as e:
            print(e)
            print("如果您使用的是 macOS 建议将 pytorch 版本升级至 2.0.0 或更高版本，以支持及时清理 torch 产生的内存占用。")


def logTime():
    current_time = time.time()
    local_time = time.localtime(current_time)
    time_str = time.strftime('%Y%m%d%H%M%S', local_time)
    return time_str


def build_prompt(history):
    prompt = "欢迎使用 ChatGLM-6B 模型，输入内容即可进行对话，clear 清空对话历史，stop 终止程序"
    for query, response in history:
        prompt += f"\n\n用户：{query}"
        prompt += f"\n\nChatGLM-6B：{response}"
    return prompt


def signal_handler(signal, frame):
    global stop_stream
    stop_stream = True


def main():
    history = []
    global stop_stream
    print("欢迎使用 ChatGLM-6B 模型，输入内容即可进行对话，clear 清空对话历史，stop 终止程序")
    while True:
        query = input("\n用户：")
        if query.strip() == "stop":
            break
        if query.strip() == "clear":
            history = []
            os.system(clear_command)
            print("欢迎使用 ChatGLM-6B 模型，输入内容即可进行对话，clear 清空对话历史，stop 终止程序")
            continue
        count = 0
        for response, history in model.stream_chat(tokenizer, query, history=history):
            if stop_stream:
                stop_stream = False
                break
            else:
                count += 1
                if count % 8 == 0:
                    os.system(clear_command)
                    print(build_prompt(history), flush=True)
                    signal.signal(signal.SIGINT, signal_handler)
        os.system(clear_command)
        print(build_prompt(history), flush=True)


def get_path_list():
    mav_list = os.listdir(DATA_PATH)
    say_list =[f"{DATA_PATH}/{i}" for i in mav_list if os.path.exists(f"{DATA_PATH}/{i}")]
    return say_list


def get_say(data_path):

    with open(data_path, "r", encoding='utf-8') as f:#, errors='ignore'
        lines = f.readlines()

    dialog_list = lines[1:]
    result = '' #[]
    cur_role = ""
    cur_dialog = ""
    for i, line in enumerate(dialog_list):
        segs = line.strip().split("|")
        cur_role = segs[1]

        if cur_dialog != "":
            cur_dialog += segs[4]
        else:
            cur_dialog = segs[4]

        if i + 1 < len(dialog_list):
            next_role = dialog_list[i + 1].split("|")[1]

            if next_role != cur_role:
                result += f"{cur_dialog} \n"
                cur_role = ""
                cur_dialog = ""
            else:
                continue
        else:
            result += f"{cur_dialog} \n"
            cur_role = ""
            cur_dialog = ""

    return result


def main2():
    history = []
    global stop_stream
    #print("欢迎使用 ChatGLM-6B 模型，输入内容即可进行对话，clear 清空对话历史，stop 终止程序")
    say_path_list = get_path_list()
    print('对话总通数：',len(say_path_list))

    t1 = time.time()
    #遍历每个对话进行处理
    q_r_dict = {}
    for path in tqdm(say_path_list):
        print(f"当前处理文件：{path}")

        if path.split('/')[-1] in EXCEPTION_FILES:
            print(f"skip file: {path}")
            continue

        try:
            customer_say = get_say(path)
            # query = '：请基于以下客服与客户对话，进行总结任务——客户关心的主题，总结字数在10个字以内，返回格式：主题：对应主题 对话如下："""" {}"""" '.format(costomer_say)
            # model.stream_chat(tokenizer, query, history=history)
            
            query = PROMPT_TEMPLATE.format(customer_say = customer_say)

            #print('query',query)
            response, history = model.chat(tokenizer, query, history=[])
            print('     response: ', response)
            q_r_dict[response] = query

            torch_gc(DEVICE)

        except Exception as e:
            df = pd.DataFrame([q_r_dict]).T
            df = df.reset_index()
            df.columns = ['主题', '原对话']
            file_path = f"./outputs/exception_topictask_{logTime()}.xlsx"
            df.to_excel(file_path)

            print(f"exception: {e}")
            print(f"exception while processing file：{path}")

            torch_gc(DEVICE)


    #保存结果xlsx
    df = pd.DataFrame([q_r_dict]).T
    df = df.reset_index()
    df.columns = ['主题', '原对话']
    file_path = f"./outputs/topictask_{logTime()}.xlsx"
    df.to_excel(file_path)
    print('保存成功！！')

    t2 = time.time()

    #print(q_r_dict)
    print('耗时秒数', t2 - t1)


## 多进程
# def data_on_processing(data_list, i, tasks_dict):
#     print(f"Task {i} start!")

#     # print(f"data_on_processing| name:{__name__}, model: {id(model)}, pid: {os.getpid()}")

#     for path in tqdm(data_list):
#         print(f"当前处理文件：{path}")

#         if path.split('/')[-1] in EXCEPTION_FILES:
#             print(f"skip file: {path}")
#             continue

#         try:
#             costomer_say = get_say(path)
#             # query = '：请基于以下客服与客户对话，进行总结任务——客户关心的主题，总结字数在10个字以内，返回格式：主题：对应主题 对话如下："""" {}"""" '.format(costomer_say)
#             # model.stream_chat(tokenizer, query, history=history)
            
#             query = PROMPT_TEMPLATE.format(costomer_say)

#             #print('query',query)
#             response, history = model.chat(tokenizer, query, history=[])
#             print('response ===',response)
#             if response is not None:
#                 tasks_dict[response] = query

#             torch_gc(DEVICE)

#         except Exception as e:
#             print(f"exception while processing file：{path}")
#             torch_gc(DEVICE)


#     print(f"Task {i} done!")


## 多进程
# def data_did_process(data_count, tasks_dict):

#     # print(f"data_did_process| name:{__name__}, model: {id(model)}, pid: {os.getpid()}")
    
#     start_time = time.time()

#     while True:
#         if len(tasks_dict) >= data_count:
#             break
#         time.sleep(0.1)


#     #保存结果xlsx
#     result_dict = dict(tasks_dict)
#     df = pd.DataFrame([result_dict]).T
#     df = df.reset_index()
#     df.columns = ['主题', '原对话']
#     file_path = f"./outputs/对话主题总结_{logTime()}.xlsx"
#     df.to_excel(file_path)
#     print('保存成功！！')

#     end_time = time.time()

#     #print(q_r_dict)
#     print('All Done! 耗时秒数', end_time - start_time)


## 多进程
# def async_main2():
#     # 注意：Unix中进程创建方式模型不是spawn，而是fork，当使用spwan时，
#     # 子进程的__name__会变成__mp_main__，而不是__main__，所以需要在子进程中重新导入模块
#     # 且写在函数外的全局变量，子进程无法访问，需要在子进程中重新定义（即不同的对象，因为子进程拥有自己的独立的内存空间）
#     mp.set_start_method('spawn')   #  To use CUDA with multiprocessing, you must use the 'spawn' start method

#     history = []
#     global stop_stream
#     #print("欢迎使用 ChatGLM-6B 模型，输入内容即可进行对话，clear 清空对话历史，stop 终止程序")
#     say_path_list = get_path_list()
#     data_count = len(say_path_list)
#     print('对话总通数：', data_count)

#     tasks_dict = mp.Manager().dict()
#     num_workers = 2
#     processes = []
#     segment_size = int(data_count / num_workers)
#     for i in range(num_workers):
#         if i == num_workers - 1:
#             segment = say_path_list[i * segment_size:]
#         else:
#             segment = say_path_list[i * segment_size: (i + 1) * segment_size]
#         p = mp.Process(target = data_on_processing, args = (segment, i, tasks_dict))
#         processes.append(p)

#     [p.start() for p in processes]

#     result_process = mp.Process(target = data_did_process, args = (data_count, tasks_dict))
#     result_process.start()
#     result_process.join()



if __name__ == "__main__":
    main2()
    # async_main2()
