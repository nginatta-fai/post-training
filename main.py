from datasets import load_dataset
from helper import load_model_and_tokenizer, test_model_with_questions
from trl import SFTConfig, SFTTrainer

def SFT(model_name, dataset_path, questions, USE_GPU):
    model, tokenizer = load_model_and_tokenizer(model_name, USE_GPU)

    train_dataset = load_dataset(dataset_path)['train']
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

def main():
    SFT(model_name="Qwen/Qwen3-0.6B-Base",
        dataset_path="banghua/DL-SFT-Dataset",
        questions = [
            "Give me an 1-sentence introduction of LLM.",
            "Calculate 1+1-1",
            "What's the difference between thread and process?"
        ],
        USE_GPU=False
    )

if __name__ == "__main__":
    main()
