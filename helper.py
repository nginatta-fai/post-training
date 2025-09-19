import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def generate_response(model, tokenizer, user_message, system_message=None, max_token=100):
    messages = []
    if system_message:
        messages.append({"role": "system", "content": system_message})

    messages.append({"role": "user", "content": user_message})

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        generation_output = model.generate(
            **inputs,
            max_new_tokens=max_token,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
    )
        
    input_length = inputs.input_ids.shape[1]
    generated_text = tokenizer.decode(generation_output[0][input_length:], skip_special_tokens=True)
    return generated_text

def test_model_with_questions(model, tokenizer, questions, 
                              system_message=None, title="Model Output"):
    print(f"\n=== {title} ===")
    for i, question in enumerate(questions, 1):
        response = generate_response(model, tokenizer, question, 
                                      system_message)
        print(f"\nModel Input {i}:\n{question}\nModel Output {i}:\n{response}\n")

def load_model_and_tokenizer(model_name, USE_GPU=False):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    if USE_GPU:
        model = model.to("cuda")

    if not tokenizer.chat_template:
        tokenizer.chat_template = """{% for message in messages %}
                {% if message['role'] == 'system' %}System: {{ message['content'] }}\n
                {% elif message['role'] == 'user' %}User: {{ message['content'] }}\n
                {% elif message['role'] == 'assistant' %}Assistant: {{ message['content'] }} <|endoftext|>
                {% endif %}
                {% endfor %}"""
    
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer
