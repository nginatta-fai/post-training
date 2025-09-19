import sft

def main():
    sft.run(
        model_name="Qwen/Qwen3-0.6B-Base",
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
