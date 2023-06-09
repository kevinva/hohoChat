import torch.cuda
import torch.backends
import os

embedding_model_dict = {
    "ernie-tiny": "nghuyong/ernie-3.0-nano-zh",
    "ernie-base": "nghuyong/ernie-3.0-base-zh",
    "text2vec-base": "shibing624/text2vec-base-chinese",
    "text2vec": "GanymedeNil/text2vec-large-chinese",
}

# Embedding model name
EMBEDDING_MODEL = "text2vec"

# Embedding running device
EMBEDDING_DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# supported LLM models
llm_model_dict = {
    "chatyuan": "ClueAI/ChatYuan-large-v2",
    "chatglm-6b-int4-qe": "THUDM/chatglm-6b-int4-qe",
    "chatglm-6b-int4": "THUDM/chatglm-6b-int4",
    "chatglm-6b-int8": "THUDM/chatglm-6b-int8",
    "chatglm-6b": "THUDM/chatglm-6b",
}

# LLM model name
LLM_MODEL = "chatglm-6b-int4"

# LLM streaming reponse
STREAMING = False

# Use p-tuning-v2 PrefixEncoder
USE_PTUNING_V2 = False

# LLM running device
LLM_DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

VS_ROOT_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "vector_store")

UPLOAD_ROOT_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "content")

API_UPLOAD_ROOT_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "api_content")

# 基于上下文的prompt模版，请务必保留"{question}"和"{context}"
# PROMPT_TEMPLATE = """已知信息：
# {context} 

# 根据上述已知信息，简洁和专业的来回答用户的问题。如果无法从中得到答案，请说 “根据已知信息无法回答该问题” 或 “没有提供足够的相关信息”，不允许在答案中添加编造成分，答案请使用中文。 问题是：{question}"""


# 关于法律助手的prompt
# PROMPT_TEMPLATE = """
# 你是一个法律助手，工作是根据已知信息专业的回答用户的法律问题。如果你对答案不确定，请说 “我不知道”，并提供一个网址引导用户去查询更多信息。 

# ---
# 已知信息：{context}
# ---


# ---
# 问题是：{question}
# ---

# 回答：
# """

PROMPT_TEMPLATE = """
作为法律助手，你的任务是依据已知信息专业地回答用户的法律问题。如果你对答案不确定，请回答 "我不确定" ，并提供一个网址引导用户查询更多信息。

请回答以下问题，并提供细节丰富、准确的回答。如果需要，你可以基于已知信息提出进一步的问题，以便更好地理解用户的需求。

请注意回答问题时，尽可能提供具有可操作性的建议，顾虑到用户需要的实际情况。如果需要，你可以简化法律术语，使答案更易于理解。

已知信息：{context}
问题：{question}

回答：

"""


# 匹配后单段上下文长度
CHUNK_SIZE = 500
