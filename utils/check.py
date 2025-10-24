import torch

def check_model_dtype(model_or_state_dict):
    is_fp16 = True
    is_fp32 = True

    if isinstance(model_or_state_dict, torch.nn.Module):
        params = model_or_state_dict.parameters()
    else:  # 假设是 state_dict
        params = model_or_state_dict.values()

    for param in params:
        if param.dtype == torch.float16:
            is_fp32 = False
        elif param.dtype == torch.float32:
            is_fp16 = False

    if is_fp16 and not is_fp32:
        print("Model is fp16.")
    elif is_fp32 and not is_fp16:
        print("Model is fp32.")
    else:
        print("Model has mixed precision.")

model = torch.load('./checkpoints/gesture/fp32.pt')
check_model_dtype(model)
for name, param in model.items() if isinstance(model, dict) else model.named_parameters():
    print(param[0])
    print(f"Parameter: {name}, dtype: {param.dtype}")
    break
model = torch.load('./checkpoints/gesture/fp16.pt')
check_model_dtype(model)
for name, param in model.items() if isinstance(model, dict) else model.named_parameters():
    print(param[0])
    print(f"Parameter: {name}, dtype: {param.dtype}")
    break
