import time
import os, sys
import json

langchain_ChatGLM_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "langchain-ChatGLM-old")
sys.path.append(langchain_ChatGLM_path)

from configs import model_config
from models.chatglm_llm import ChatGLM
from chains.local_doc_qa import similarity_search_with_score_by_vector, LLM_HISTORY_LEN, get_docs_with_score 


LLM_MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "models", "chatglm-6b-int4")


def init_llm(local_path = None):
    start_time = time.time()

    llm = ChatGLM()

    if local_path is not None:
        llm.load_model(model_name_or_path = local_path,
                       llm_device = model_config.LLM_DEVICE,
                       use_ptuning_v2 = model_config.USE_PTUNING_V2)
    else:
        llm.load_model(model_name_or_path = model_config.llm_model_dict[model_config.LLM_MODEL],
                       llm_device = model_config.LLM_DEVICE,
                       use_ptuning_v2 = model_config.USE_PTUNING_V2)
    llm.history_len = LLM_HISTORY_LEN

    return llm


g_llm = init_llm(local_path = LLM_MODEL_PATH)


def answer_question(prompt, raw_query = "", chat_history = []):
    final_result = None

    for result, history in g_llm._call(prompt = prompt, history = chat_history, streaming = False):
        history[-1][0] = raw_query
        final_result = result

    return final_result, history

def data_preprocess(data_path):
    with open(data_path, "r") as f:
        lines = f.readlines()

    dialog_list = lines[1:]
    result = []
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

            if  next_role != cur_role:
                print(f"{cur_role}:{cur_dialog}")
                result.append(f"{cur_role}:{cur_dialog}")
                cur_role = ""
                cur_dialog = ""
            else:
                continue
        else:
            print(f"{cur_role}:{cur_dialog}")
            result.append(f"{cur_role}:{cur_dialog}")
            cur_role = ""
            cur_dialog = ""


    return result


def main():
    data_path = "./0831315.wav_raw.txt"
    result_list = data_preprocess(data_path)
    text = "\n".join(result_list)
    
    prompt = f"""
    请基于以下对话，进行主题总结任务，总结字数在10个字以内，返回格式：主题：对应主题 对话：'{text}'
    """
    result, history = answer_question(prompt = prompt, raw_query = prompt)
    print(f"prompt: {prompt}, result: {result}")


if __name__ == "__main__":
    main()