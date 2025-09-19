from datasets import load_dataset
from helper import load_model_and_tokenizer, test_model_with_questions, generate_response
from trl import SFTConfig, SFTTrainer, DPOConfig, DPOTrainer

def SFT(model_name, dataset_path, questions, USE_GPU):
    model, tokenizer = load_model_and_tokenizer(model_name, USE_GPU)

    train_dataset = load_dataset(dataset_path, split="train")
    if not USE_GPU:
        train_dataset=train_dataset.select(range(150))

    sft_config = SFTConfig(
        learning_rate=8e-5,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        gradient_checkpointing=False,
        logging_steps=2
    )

    sft_trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        processing_class=tokenizer
    )

    sft_trainer.train()

    if not USE_GPU: # move model to CPU when GPU isn’t requested
        sft_trainer.model.to("cpu")
    
    test_model_with_questions(sft_trainer.model, tokenizer, questions, 
                          title="Base Model (After SFT) Output")

def DPO(model_name, dataset_path, questions, USE_GPU):
    model, tokenizer = load_model_and_tokenizer(model_name, USE_GPU)
    train_dataset = load_dataset(dataset_path, split="train")

    if not USE_GPU:
        train_dataset=train_dataset.select(range(150))

    #
    # USE THIS CODE IF YOU WANT TO BUILD A SPECIIFC IDENTITY DATASET FROM A GENERIC ONE
    #
    # NEW_NAME = "Deep Qwen"
    # ORIG_NAME = "Qwen"
    # SYSTEM_PROMPT = f"You are a helpful assistant"

    # if not USE_GPU:
    #     dataset=dataset.select(range(50))

    # def build_dpo_chatml(example):
    #     messages = example["conversations"]
    #     prompt = next(m["value"] for m in messages if m["from"] == "human")

    #     try:
    #         rejected_response = generate_response(model, tokenizer, prompt)
    #     except Exception as e:
    #         rejected_response = "Error: Failed to generate response"
    #         print(f"Generation error for pormpt: {prompt}\nError: {e}")

    #     accepted_response = rejected_response.replace(ORIG_NAME, NEW_NAME)

    #     accepted = [
    #         {"role": "system", "content": SYSTEM_PROMPT},
    #         {"role": "user", "content": prompt},
    #         {"role": "assistant", "content": accepted_response}
    #     ]

    #     rejected = [
    #         {"role": "system", "content": SYSTEM_PROMPT},
    #         {"role": "user", "content": prompt},
    #         {"role": "assistant", "content": rejected_response}
    #     ]

    #     return {"accepted": accepted, "rejected": rejected}

    # dataset = dataset.map(build_dpo_chatml, remove_columns=dataset.column_names)

    config = DPOConfig(
        beta=0.2,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=1,
        learning_rate=5e-5,
        logging_steps=2,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=config,
        train_dataset=train_dataset,
        processing_class=tokenizer
    )

    trainer.train()

    test_model_with_questions(trainer.model, tokenizer, questions, title="Post-traineed model (After DPO) Output")


def main():
    # SFT(
    #     model_name="Qwen/Qwen3-0.6B-Base",
    #     dataset_path="banghua/DL-SFT-Dataset",
    #     questions = [
    #         "Give me an 1-sentence introduction of LLM.",
    #         "Calculate 1+1-1",
    #         "What's the difference between thread and process?"
    #     ],
    #     USE_GPU=False
    # )

    DPO(
        model_name="Qwen/Qwen2.5-0.5B-Instruct",
        dataset_path="banghua/DL-DPO-Dataset",
        questions = [
            "What is your name?",
            "Are you ChatGPT?",
            "Tell me about your name and organization."
        ],
        USE_GPU=False
    )

if __name__ == "__main__":
    main()
