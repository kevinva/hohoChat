import os
import argparse
import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer


CHECKPOINT_PATH = "output/law-chatglm-6b-int4-pt-128-2e-2/checkpoint-3000"

g_parser = argparse.ArgumentParser(description = "Help Information")
g_parser.add_argument('--query', '-q', default = '你好', help = 'input your question')
g_args = g_parser.parse_args()

def main():
    tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b-int4", trust_remote_code = True, revision = "v0.1.0")
    config = AutoConfig.from_pretrained("THUDM/chatglm-6b-int4", trust_remote_code = True, pre_seq_len = 128, revision = "v0.1.0")
    model = AutoModel.from_pretrained("THUDM/chatglm-6b-int4", config = config, trust_remote_code = True, revision = "v0.1.0")
    prefix_state_dict = torch.load(f"{CHECKPOINT_PATH}/pytorch_model.bin")

    # print(f"hoho: prefix_state_dict: {prefix_state_dict}")
    print(f"query = {g_args.query}")

    new_prefix_state_dict = {}
    for k, v in prefix_state_dict.items():
        if k.startswith("transformer.prefix_encoder."):
            new_prefix_state_dict[k[len("transformer.prefix_encoder."):]] = v  # 提取“transformer.prefix_encoder.”后面剩余的字符串作为key
    model.transformer.prefix_encoder.load_state_dict(new_prefix_state_dict)


    # model = model.quantize(4) # 已经是int4了，不需要再quantize了
    model= model.half().cuda()
    model.transformer.prefix_encoder.float()
    model = model.eval()

    response, history = model.chat(tokenizer, g_args.query, history = [])
    print(f'response: {response}\n history: {history}')


if __name__ == "__main__":
    # main()
    print(f'main: cuda = {torch.cuda.is_available()}, cudnn = {torch.backends.cudnn.enabled}')