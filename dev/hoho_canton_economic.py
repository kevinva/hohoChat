import time
import os, sys
import json

# 放在所有访问GPU代码之前（作为从卡，window要设置这个才行）
os.environ['CUDA_VISIBLE_DEVICES']='1' 
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


# from langchain-ChatGLM-master
from configs import model_config
from models.chatglm_llm import ChatGLM
from chains.local_doc_qa import similarity_search_with_score_by_vector, LLM_HISTORY_LEN, get_docs_with_score

# from tools
from hoho_utils import logTime, strForTime_YmdHMS
from hoho_huggingface import HohoHuggingFaceEmbeddings


LOG_PREFIX = "[CAN_ECONOMIC]"
DOCS_DATA_PATH = r"F:\AI\hohoChat\data\can_economic.pdf"
LLM_MODEL_PATH = r"H:\AI\LLMs\chatglm2-6b-int4"
EMBEDDING_MODEL_PATH = r"H:\AI\embedding_models\multi-qa-mpnet-base-dot-v1"
PROMPT_TEMPLATE = """
你是一名数据分析师，请基于你的知识从以下文本中提取指标名称、对应的指标值、数值单位和数据期，结果以json形式返回。
如果结果有多个，则以jsonlines形式返回。
如果你对答案不确定，请说 “我不知道”。
文本：{content}
结果：
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

    text_splitter = TokenTextSplitter(chunk_size = 250 , chunk_overlap = 0, disallowed_special = ())
    docs = []
    loader = UnstructuredPDFLoader(docs_path)
    print(f"loader: {loader}")
    docs += loader.load_and_split(text_splitter)

    print(f"{LOG_PREFIX}[{logTime()}] Number of docs: {len(docs)}")
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


def get_answer(content):
    start_time = time.time()

    prompt = PROMPT_TEMPLATE.format(content = content)
    print(f'{LOG_PREFIX} [get_answer] prompt: {prompt}')

    final_result = None
    for result, history in g_llm._call(prompt = prompt, history = [], streaming = False):
        final_result = result

    return final_result


def main():
    for i, doc in enumerate(g_docs):
        print(doc)
        if i > 10:
            break


if __name__ == "__main__":
    main()