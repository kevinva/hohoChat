import time
import os, sys
import json
import re

# 放在所有访问GPU代码之前（作为从卡，window要设置这个才行）
# os.environ['CUDA_VISIBLE_DEVICES']='1' 
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"  # window要设置这个，否则会即使显存够也会出现OOM错误

langchain_ChatGLM_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "langchain-ChatGLM-master")
tools_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools")
sys.path.append(langchain_ChatGLM_path)
sys.path.append(tools_path)


from langchain.text_splitter import CharacterTextSplitter, TokenTextSplitter
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import UnstructuredPDFLoader, DirectoryLoader, TextLoader
from langchain.schema import Document


# langchain-ChatGLM-master模块
from configs import model_config
from models.chatglm_llm import ChatGLM
from chains.local_doc_qa import similarity_search_with_score_by_vector, LLM_HISTORY_LEN, get_docs_with_score

# tools模块
from hoho_utils import logTime, strForTime_YmdHMS
from hoho_huggingface import HohoHuggingFaceEmbeddings


LOG_PREFIX = "[CAN_ECONOMIC]"
DOCS_DATA_PATH = "../data/test_doc.pdf"
LLM_MODEL_PATH = "../models/chatglm2-6b-int4"
EMBEDDING_MODEL_PATH = "../models/multi-qa-mpnet-base-dot-v1"

PROMPT_TEMPLATE = """
###{content}###

从上文中，提取"指标名称"、"指标值"、"数值单位"、"数据期"等类型的实体，以json列表形式输出提取结果。
"""


def init_knowledge_based(docs_path = None, embedding_model_path = None):
    start_time = time.time()

    if embedding_model_path is not None:
        embedding_model_name = os.path.basename(embedding_model_path)
        embeddings = HohoHuggingFaceEmbeddings(model_name = embedding_model_path,
                                               cache_folder = embedding_model_path,
                                               model_kwargs = {'device': model_config.EMBEDDING_DEVICE})
    else:
        embedding_model_name = model_config.EMBEDDING_MODEL
        embeddings = HuggingFaceEmbeddings(model_name = model_config.embedding_model_dict[model_config.EMBEDDING_MODEL],
                                           model_kwargs = {'device': model_config.EMBEDDING_DEVICE})

    if docs_path is None:
        print(f"{LOG_PREFIX}[{logTime()}] docs_path is None!")
        return None

    loader = UnstructuredPDFLoader(docs_path)
    data = loader.load()
    print(f"{LOG_PREFIX}[{logTime()}] Document's len: {len(data[0].page_content)}")

    text_splitter = TokenTextSplitter(chunk_size = 2000, chunk_overlap = 0, disallowed_special = ())
    docs = []
    docs += loader.load_and_split(text_splitter)

    print(f"{LOG_PREFIX}[{logTime()}] Number of chunks: {len(docs)}")
    print(f"{LOG_PREFIX}[{logTime}] Load knowledge done! Elapsed time: {time.time() - start_time} seconds")

    return docs


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

    print(f"{LOG_PREFIX}[{logTime()}] Load llm done! Elapsed time: {time.time() - start_time} seconds")

    return llm


print(f"{LOG_PREFIX}[{logTime()}] Initializing knowledge...", flush = True)
g_docs = init_knowledge_based(docs_path = DOCS_DATA_PATH, embedding_model_path = EMBEDDING_MODEL_PATH)
print(f"{LOG_PREFIX}[{logTime()}] Initializing knowledge successfully!", flush = True)

print(f"{LOG_PREFIX}[{logTime()}] Initializing llm...", flush = True)
g_llm = init_llm(local_path = LLM_MODEL_PATH)
print(f"{LOG_PREFIX}[{logTime()}] Initializing llm successfully!", flush = True)



def contains_arabic_numbers(text):
    pattern = r"\d"
    match = re.search(pattern, text)
    return match is not None


def get_answer(query):
    start_time = time.time()

    prompt = PROMPT_TEMPLATE.format(content = query)

    final_result = None
    for result, history in g_llm._call(prompt = prompt, history = [], streaming = False):
        final_result = result

    print(f"{LOG_PREFIX}[{logTime()}] Elapsed time: {time.time() - start_time} seconds")

    return final_result


def main():
    result_list = []
    for i, doc in enumerate(g_docs):
        print(f"###### doc {i} ######")
        if contains_arabic_numbers(doc.page_content):
            result_list.append(get_answer(doc.page_content))
        else:
            print("skip!")
    
    output_file = "../outputs/result_json.txt"
    result_str = "\n\n\n\n\n\n".join(result_list)
    with open(output_file, "w", encoding = "utf-8") as f:
        f.write(result_str)
    

if __name__ == "__main__":
    main()