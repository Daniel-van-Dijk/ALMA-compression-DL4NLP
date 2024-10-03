# from https://github.com/MrYxJ/calculate-flops.pytorch/blob/main/calflops/utils.py
def generate_transformer_input(model_tokenizer, input_shape, device):
    """Automatically generates data in the form of transformes model input format.
    
    Args:
        input_shape (tuple):transformers model input shape: (batch_size, seq_len).
        tokenizer (transformer.model.tokenization): transformers model tokenization.tokenizer.

    Returns:
        dict: data format of transformers model input, it is a dict contain 'input_ids', 'attention_mask', sometime contain 'token_type_ids'.
    """

    if input_shape is None:
        input_shape = [1, 128]  # defautl (batch_size=1, seq_len=128)

    max_length = input_shape[1]
    model_input_ids = []
    model_attention_mask = []
    model_token_type_ids = []
    model_position_ids = []

    inp_seq = ""
    for _ in range(input_shape[0]):
        inputs = model_tokenizer.encode_plus(
            inp_seq,
            add_special_tokens=True,
            truncation_strategy='longest_first',
        )
        origin_length = len(inputs["input_ids"])
        padding_length = max_length - origin_length

        for key in inputs.keys():
            if key == "input_ids":
                input_ids = inputs["input_ids"]
                pad_token = model_tokenizer.pad_token_id if model_tokenizer.pad_token_id else 0
                input_ids = input_ids + ([pad_token] * padding_length)
                assert len(input_ids) == max_length, "len(input_ids) must equal max_length"
                model_input_ids.append(input_ids)
            elif key == "attention_mask":
                attention_mask = [1] * origin_length
                attention_mask = attention_mask + ([0] * padding_length)
                assert len(attention_mask) == max_length, "len(attention_mask) must equal max_length"
                model_attention_mask.append(attention_mask)
            elif key == "token_type_ids":
                token_type_ids = inputs['token_type_ids']
                pad_token_segment_id = 0
                token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)
                assert len(token_type_ids) == max_length, "len(token_type_ids) must equal max_length"
                model_token_type_ids.append(token_type_ids)
            elif key == "position_ids":  # chatglm2 use position id
                position_ids = inputs['position_ids']
                for i in range(origin_length, max_length):
                    position_ids.append(i)
                assert len(position_ids) == max_length, "len(position_ids) must equal max_length"
                model_position_ids.append(position_ids)

    # Batch size input_shape[0], sequence length input_shape[128]
    inputs = {}
    if len(model_input_ids) > 0:
        inputs.update({"input_ids": torch.tensor(model_input_ids).to(device)})
    if len(model_attention_mask) > 0:
        inputs.update({"attention_mask": torch.tensor(model_attention_mask).to(device)})
    if len(model_token_type_ids) > 0:
        inputs.update({'token_type_ids': torch.tensor(model_token_type_ids).to(device)})
    if len(model_position_ids) > 0:
        inputs.update({'position_ids': torch.tensor(model_position_ids).to(device)})

    return inputs
