import os
import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer 

CHECKPOINT_PATH = "output/adgen-chatglm-6b-pt-128-2e-2/checkpoint-3000"

tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6B-int4", trust_remote_code = True)
config = AutoConfig.from_pretrained("THUDM/chatglm-6B-int4", trust_remote_code = True, pre_seq_len = 128)
model = AutoModel.from_pretrained("THUDM/chatglm-6B-int4", config = config, trust_remote_code = True)
prefix_state_dict = torch.load(os.path.join(CHECKPOINT_PATH, "pytorch_model.bin"))

print(f"hoho: prefix_state_dict: {prefix_state_dict}")

new_prefix_state_dict = {}
for k, v in prefix_state_dict.items():
    if k.startswith("transformer.prefix_encoder."):
        new_prefix_state_dict[k[len("transformer.prefix_encoder."):]] = v
model.transformer.prefix_encoder.load_state_dict(new_prefix_state_dict)


# model = model.quantize(4) # 已经是int4了，不需要再quantize了？
model= model.half().cuda()
model.transformer.prefix_encoder.float()
model = model.eval()

response, history = model.chat(tokenizer, "你好", history = [])
print(f'response: {response}\n history: {history}')
