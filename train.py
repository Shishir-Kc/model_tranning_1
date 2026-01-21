import torch
from datasets import load_dataset
from decouple import config

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model


MODEL_ID ="google/functiongemma-270m-it"



tokenizer = AutoTokenizer.from_pretrained(MODEL_ID,
                                          token=config("token"))
tokenizer.pad_token = tokenizer.eos_token  # IMPORTANT

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float32,
    device_map="cpu"
)

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()


dataset = load_dataset("json", data_files="train.jsonl")

def tokenize(example):
    texts = [tokenizer.apply_chat_template(msgs, tokenize=False) for msgs in example["messages"]]
    return tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=128
    )

tokenized_dataset = dataset.map(tokenize, batched=True, remove_columns=["messages"])


training_args = TrainingArguments(
    output_dir="./neo-lora",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=5,
    learning_rate=2e-4,
    logging_steps=1,
    save_strategy="epoch",
    fp16=False,
    bf16=False,
    report_to="none"
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    data_collator=DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
)

trainer.train()


model.save_pretrained("neo-lora-adapter")
tokenizer.save_pretrained("neo-lora-adapter")

print("✅ Training complete. LoRA adapter saved.")
