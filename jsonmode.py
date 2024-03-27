import argparse
import torch
import json

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)

from validator import validate_json_data

from utils import (
    print_nous_text_art,
    inference_logger,
    get_assistant_message,
    get_chat_template,
    validate_and_extract_tool_calls
)

# create your pydantic model for json object here
from typing import List, Optional
from pydantic import BaseModel

class Character(BaseModel):
    name: str
    species: str
    role: str
    personality_traits: Optional[List[str]]
    special_attacks: Optional[List[str]]

    class Config:
        schema_extra = {
            "additionalProperties": False
        }

# serialize pydantic model into json schema
pydantic_schema = Character.schema_json()

class ModelInference:
    def __init__(self, model_path, chat_template, load_in_4bit):
        inference_logger.info(print_nous_text_art())
        self.bnb_config = None

        if load_in_4bit == "True":
            self.bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            return_dict=True,
            quantization_config=self.bnb_config,
            torch_dtype=torch.float16,
            attn_implementation="flash_attention_2",
            device_map="auto",
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        if self.tokenizer.chat_template is None:
            print("No chat template defined, getting chat_template...")
            self.tokenizer.chat_template = get_chat_template(chat_template)
        
        inference_logger.info(self.model.config)
        inference_logger.info(self.model.generation_config)
        inference_logger.info(self.tokenizer.special_tokens_map)
    
    def run_inference(self, prompt):
        inputs = self.tokenizer.apply_chat_template(
            prompt,
            add_generation_prompt=True,
            return_tensors='pt'
        )

        tokens = self.model.generate(
            inputs.to(self.model.device),
            max_new_tokens=1500,
            temperature=0.8,
            repetition_penalty=1.1,
            do_sample=True,
            eos_token_id=self.tokenizer.eos_token_id
        )
        completion = self.tokenizer.decode(tokens[0], skip_special_tokens=False, clean_up_tokenization_space=True)
        return completion

    def generate_json_completion(self, query, chat_template, max_depth=5):
        try:
            depth = 0
            sys_prompt = f"You are a helpful assistant that answers in JSON. Here's the json schema you must adhere to:\n<schema>\n{pydantic_schema}\n<schema>"
            prompt = [{"role": "system", "content": sys_prompt}]
            prompt.append({"role": "user", "content": query})

            inference_logger.info(f"Running inference to generate json object for pydantic schema:\n{json.dumps(json.loads(pydantic_schema), indent=2)}")
            completion = self.run_inference(prompt)

            def recursive_loop(prompt, completion, depth):
                nonlocal max_depth

                assistant_message = get_assistant_message(completion, chat_template, self.tokenizer.eos_token)

                tool_message = f"Agent iteration {depth} to assist with user query: {query}\n"
                if assistant_message is not None:
                    validation, json_object, error_message = validate_json_data(assistant_message, json.loads(pydantic_schema))
                    if validation:
                        inference_logger.info(f"Assistant Message:\n{assistant_message}")
                        inference_logger.info(f"json schema validation passed")
                        inference_logger.info(f"parsed json object:\n{json.dumps(json_object, indent=2)}")
                    elif error_message:
                        inference_logger.info(f"Assistant Message:\n{assistant_message}")
                        inference_logger.info(f"json schema validation failed")
                        tool_message += f"<tool_response>\nJson schema validation failed\nHere's the error stacktrace: {error_message}\nPlease return corrrect json object\n<tool_response>"
                        
                        depth += 1
                        if depth >= max_depth:
                            print(f"Maximum recursion depth reached ({max_depth}). Stopping recursion.")
                            return
                        
                        prompt.append({"role": "tool", "content": tool_message})
                        completion = self.run_inference(prompt)
                        recursive_loop(prompt, completion, depth)
                else:
                    inference_logger.warning("Assistant message is None")
            recursive_loop(prompt, completion, depth)
        except Exception as e:
            inference_logger.error(f"Exception occurred: {e}")
            raise e

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run json mode completion")
    parser.add_argument("--model_path", type=str, help="Path to the model folder")
    parser.add_argument("--chat_template", type=str, default="chatml", help="Chat template for prompt formatting")
    parser.add_argument("--load_in_4bit", type=str, default="False", help="Option to load in 4bit with bitsandbytes")
    parser.add_argument("--query", type=str, default="Please return a json object to represent Goku from the anime Dragon Ball Z?")
    parser.add_argument("--max_depth", type=int, default=5, help="Maximum number of recursive iteration")
    args = parser.parse_args()

    # specify custom model path
    if args.model_path:
        inference = ModelInference(args.model_path, args.chat_template, args.load_in_4bit)
    else:
        model_path = 'NousResearch/Hermes-2-Pro-Mistral-7B'
        inference = ModelInference(model_path, args.chat_template, args.load_in_4bit)
        
    # Run the model evaluator
    inference.generate_json_completion(args.query, args.chat_template, args.max_depth)
