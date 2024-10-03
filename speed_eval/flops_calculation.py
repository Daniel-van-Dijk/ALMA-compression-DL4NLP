from calflops import calculate_flops
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "haoranxu/ALMA-7B-R"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device).half()

print(f"on cuda: {next(model.parameters()).is_cuda}")

def attn_flops_layer(layer, seq_length):
    # q: (num_heads, seq_length, dim_head)
    # k: (num_heads, seq_length, dim_head)
    # v: (num_heads, seq_length, dim_head)
    # q@k.T: (num_heads, seq_length, seq_length)
    # softmax(q@k.T)@V: (num_heads, seq_length, dim_head) 
    self_attn = layer.self_attn # LlamaSdpaAttention
    num_heads = self_attn.num_heads
    dim_head = self_attn.head_dim
    return 4*num_heads*seq_length*seq_length*dim_head

def get_attn_flops(model, seq_length: int):
    count = 0
    for layer in model.model.layers:
        count += attn_flops_layer(layer, seq_length)
    print(f"Attn FLOPs = {count/1e12} TFLOPs")
    return count/1e12 # return in tflops

for max_seq_length in [64,128,256,512,1024,2048,4096]:

    batch_size = 1

    print(f'\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n--- Starting calculation flops for seq_len={max_seq_length} ---')

    flops, macs, params = calculate_flops(model=model, input_shape=(batch_size, max_seq_length), transformer_tokenizer=tokenizer) # default input shape: (1, 128)
    

    flops, unit = flops.split()
    flops = float(flops)
    if unit == 'GFLOPS':
        flops /= 1000
    if unit == 'MFLOPS':
        flops /= 1000000
    linear_flops = flops
    attn_flops = get_attn_flops(model, max_seq_length)
    flops += attn_flops

    A100_flops = 312

    print(f"Expected runtime on A100 for (batch_size={batch_size}, max_seq_length={max_seq_length}): {flops/A100_flops*1000} ms \n")
    
    print(f"{model_name} FLOPs:{flops} TFLOPS  MACs:{macs}  Params:{params} \n")
    print(f"sqr_flops: {attn_flops} TFLOPS,  linear_flops: {linear_flops} TFLOPS\n")