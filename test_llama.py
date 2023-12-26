from transformers import PreTrainedModel, LlamaForCausalLM, LlamaTokenizer

tokenizer = LlamaTokenizer.from_pretrained('checkpoints/stage1')
model = LlamaForCausalLM.from_pretrained('checkpoints/stage1')

tokenizer.add_tokens(['<speech>', '<speech/>'])
model.resize_token_embeddings(len(tokenizer))

Text_Format = (
    "###[Human]:{instruction}\n\n\n"
    "###[Assistant]:"
)

def process_dataset(tokenizer, instruction):
    input = Text_Format.format(instruction=instruction)
    input_ids = tokenizer(input, return_tensors="pt").input_ids
    return input_ids

instruction = 'hello <speech> <speech> <speech/> <speech>'
import pdb
pdb.set_trace()
input_ids = process_dataset(tokenizer, instruction)
generate_ids = model.generate(input_ids)
print(tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0])
