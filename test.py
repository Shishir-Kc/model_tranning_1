from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from decouple import config

BASE_MODEL = "google/functiongemma-270m-it"
LORA_PATH = "neo-lora-adapter"


tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, token=config("token"))
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, token=config("token"))
model = PeftModel.from_pretrained(model, LORA_PATH)

while True:

    messages = [
    {"role": "user", "content": f"{input(">")}"}
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = tokenizer(prompt, return_tensors="pt")


    outputs = model.generate(
    **inputs, 
    max_new_tokens=50,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
    repetition_penalty=1.2,
    pad_token_id=tokenizer.eos_token_id
    )



    input_length = inputs.input_ids.shape[1]
    generated_tokens = outputs[0][input_length:]

    print(tokenizer.decode(generated_tokens, skip_special_tokens=True))