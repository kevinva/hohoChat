import os
import sys
import time

langchain_ChatGLM_root_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "langchain-ChatGLM-master")
sys.path.append(langchain_ChatGLM_root_path)

from IPython.display import display, Markdown, clear_output
import torch.cuda
import torch.backends


from configs import model_config
from models.chatglm_llm import ChatGLM
from chains.local_doc_qa import *
from textsplitter import ChineseTextSplitter
from utils import torch_gc

from langchain.text_splitter import MarkdownTextSplitter, CharacterTextSplitter
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import UnstructuredMarkdownLoader


DATA_DIR = "/root/hoho/data/Laws-master/"
LLM_MODEL_PATH = "/root/hoho/models/chatglm-6b-int4/"


def time_str_YmdHmS():
    current_time = time.time()
    local_time = time.localtime(current_time)
    time_str = time.strftime('%Y%m%d%H%m%S', local_time)
    return time_str


def get_filepaths_at_path(item_path):
    if os.path.isfile(item_path):
        return [item_path]
    
    result_list = []
    for item in os.listdir(item_path):
        path = os.path.join(item_path, item)
        file_paths = get_filepaths_at_path(path)
        result_list.extend(file_paths)
    
    return result_list


def init_vector_store():
    file_paths = get_filepaths_at_path(DATA_DIR)
    file_paths = [file_path for file_path in file_paths if os.path.basename(file_path) != '_index.md']

    print(f"[hoho] Nmber of file_paths: {len(file_paths)}")

    # text_splitter = ChineseTextSplitter(pdf = False)
    text_splitter = MarkdownTextSplitter(chunk_size = model_config.CHUNK_SIZE , chunk_overlap = 100)

    docs = []
    for file_path in file_paths:
        loader = UnstructuredMarkdownLoader(file_path)
        docs += loader.load_and_split(text_splitter)

    print(f"[hoho] Number of docs: {len(docs)}")

    embeddings = HuggingFaceEmbeddings(model_name = model_config.embedding_model_dict["ernie-base"], 
                                       model_kwargs = {'device': model_config.EMBEDDING_DEVICE})
    
    vector_store = FAISS.from_documents(docs, embeddings)

    vs_path = f"/root/hoho/outputs/vector_store/law_FAISS_{time_str_YmdHmS()}"
    vector_store.save_local(vs_path)

    print(f"[hoho] vector_store saved to {vs_path}")

    FAISS.similarity_search_with_score_by_vector = similarity_search_with_score_by_vector
    vector_store.chunk_size = model_config.CHUNK_SIZE

    print("[hoho] Initial vector_store successfully!")

    return vector_store


def init_llm():
    llm = ChatGLM()
    llm.load_model(model_name_or_path = model_config.llm_model_dict[model_config.LLM_MODEL],
                llm_device = model_config.LLM_DEVICE,
                use_ptuning_v2 = model_config.USE_PTUNING_V2)
    llm.history_len = LLM_HISTORY_LEN

    print("[hoho] Initial llm successfully!")

    return llm


g_vector_store = init_vector_store()
g_llm = init_llm()


def answer_based_on_knowledge(query, history = []):
    related_docs_with_score = g_vector_store.similarity_search_with_score(query, k = VECTOR_SEARCH_TOP_K)
    related_docs = get_docs_with_score(related_docs_with_score)
    prompt = generate_prompt(related_docs, query)

    # if streaming:
    #     for result, history in self.llm._stream_call(prompt = prompt,history = chat_history):
    #         history[-1][0] = query
    #         response = {"query": query,
    #                     "result": result,
    #                     "source_documents": related_docs}
    #         yield response, history
    # else:
    for result, history in g_llm._call(prompt = prompt, history = history, streaming = False):
        history[-1][0] = query
        response = {"query": query,
                    "result": result,
                    "source_documents": related_docs}
        yield response, history


def display_answer(query, history = []):
    for resp, history in answer_based_on_knowledge(query, history):
        clear_output(wait = True)
        display(Markdown(resp['result']))
    
    return resp, history


def main():
    # question = "信用卡欠款不还会遭到什么处罚？"
    # answer, history = display_answer(question, history = [])

    print("[hoho] main()")



if __name__ == "__main__":
    main()