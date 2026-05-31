import os
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

MODEL = "unsloth/Qwen2.5-0.5B-Instruct"
model, tok = FastLanguageModel.from_pretrained(
    model_name=MODEL, max_seq_length=2048, load_in_4bit=True, dtype=None)
model = FastLanguageModel.get_peft_model(
    model, r=8, lora_alpha=16,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    use_gradient_checkpointing="unsloth", random_state=42)

ds = load_dataset("json", data_files=os.path.expanduser("~/proof.jsonl"), split="train")
def fmt(ex):
    return {"text": tok.apply_chat_template(ex["messages"], tokenize=False)}
ds = ds.map(fmt, remove_columns=ds.column_names)
print("DATASET_OK rows:", ds.num_rows)

trainer = SFTTrainer(
    model=model, tokenizer=tok, train_dataset=ds,
    args=SFTConfig(
        per_device_train_batch_size=1, gradient_accumulation_steps=4,
        max_steps=20, learning_rate=2e-4, logging_steps=2, warmup_steps=2,
        output_dir=os.path.expanduser("~/proof_out"),
        max_seq_length=2048, dataset_text_field="text",
        report_to="none", seed=42, optim="adamw_8bit"))
stats = trainer.train()
print("TRAIN_OK final_loss:", stats.training_loss)
model.save_pretrained(os.path.expanduser("~/proof_lora"))
print("ADAPTER_SAVED ~/proof_lora")
