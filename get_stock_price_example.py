from transformers import MistralForCausalLM
from transformers import LlamaTokenizer

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")

model = AutoModelForCausalLM.from_pretrained(
    'NousResearch/Nous-Hermes-2-PlusPlus',
    torch_dtype=torch.float16,
    device_map="auto",
    load_in_8bit=False,
    load_in_4bit=True,
    use_flash_attention_2=True,
    trust_remote_code=True
)


chat = [
  {"role": "system", "content": ""}, ## TODO: Add system prompt example 
  {"role": "user", "content": "I need the current stock price of Tesla (TSLA)"}
]

current_chat = tokenizer.apply_chat_template(chat, tokenize=False)

generated_chat = model.generate(current_chat, max_new_tokens=1500, temperature=0.8, repetition_penalty=1.1, do_sample=True, eos_token_id=tokenizer.eos_token_id)

response = tokenizer.decode(generated_ids[0], skip_special_tokens=True, clean_up_tokenization_space=True)

print(response)

## TODO: Parse function call

## TODO: Call Function

## TODO: Generate tool role response

## TODO: Generate model response
