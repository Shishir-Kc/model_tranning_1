from transformers import AutoTokenizer, AutoModelForCausalLM
from decouple import config

model_id = "google/functiongemma-270m-it"

tokenizer = AutoTokenizer.from_pretrained(model_id,
                                          token=config("token"))
model = AutoModelForCausalLM.from_pretrained(model_id,token=config("token"))

print("Model loaded successfully!")

prompt = " who are you ?"

inputs = tokenizer(prompt, return_tensors="pt")

outputs = model.generate(
    **inputs,
    max_new_tokens=30,
    do_sample=True,
    temperature=0.3,
    top_p=0.9,
    eos_token_id = tokenizer.eos_token_id
)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
