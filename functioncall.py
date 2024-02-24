import argparse
import torch
import json
from tqdm import tqdm
from tenacity import retry, wait_random_exponential, stop_after_attempt, RetryError

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)

import functions
from prompter import PromptManager
from validator import validate_function_call_schema

from utils import (
    eval_logger,
    get_assistant_message,
    get_chat_template,
    validate_and_extract_tool_calls
)

class ModelInference:
    def __init__(self, model_path, chat_template, load_in_4bit):
        self.prompter = PromptManager()
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
            use_flash_attention_2=True,
            device_map="auto",
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        if self.tokenizer.chat_template is None:
            self.tokenizer.chat_template = get_chat_template(chat_template)
        
        eval_logger.info(self.model.config)
        eval_logger.info(self.model.generation_config)
        eval_logger.info(self.tokenizer.chat_template)
        eval_logger.info(self.tokenizer.special_tokens_map)

    def process_completion_and_validate(self, tokens, chat_template, tools):
        completion = self.tokenizer.decode(tokens[0], skip_special_tokens=False, clean_up_tokenization_space=True)
        eval_logger.info(f"model completion with eval prompt:\n{completion}")

        assistant_message = get_assistant_message(completion, chat_template, self.tokenizer.eos_token)

        if assistant_message:
            validation, tool_calls = validate_and_extract_tool_calls(assistant_message)

            if validation and all(validate_function_call_schema(tool_call, tools) for tool_call in tool_calls):
                eval_logger.info(f"all validations passed")
                eval_logger.info(f"parsed tool calls:\n{json.dumps(tool_calls, indent=2)}")
                return tool_calls
            else:
                eval_logger.info("Validation failed for function calls")
                eval_logger.info(f"Assistant message: {assistant_message}")
                if validation is False and assistant_message is None:
                    eval_logger.warning("Validation failed due to None assistant message")
                    raise ValueError("Validation failed for function calls")
        else:
            eval_logger.warning("Assistant message is None")
            raise ValueError("Assistant message is None")
        
    def execute_function_call(self, tool_call):
        function_name = tool_call.get("name")
        function_to_call = getattr(functions, function_name, None)
        function_args = tool_call.get("arguments", {})
        function_response = function_to_call(*function_args.values())
        print(function_response)

    def generate_function_call(self, query, chat_template, num_fewshot):
        try:
            chat = [
                {"role": "user", "content": query}
            ]
            tools = functions.get_openai_tools()
            prompt = self.prompter.generate_prompt(chat, tools, num_fewshot)

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

            # Call the separate function for completion and validation
            tool_calls = self.process_completion_and_validate(tokens, chat_template, tools)

            if tool_calls:
                for tool_call in tool_calls:
                    self.execute_function_call(tool_call)

        except Exception as e:
            # Log the exception or perform any specific actions
            eval_logger.error(f"Unhandled exception occurred: {e}")
            raise e 
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model performance on fireworks-ai dataset")
    parser.add_argument("--model_path", type=str, help="Path to the model folder")
    parser.add_argument("--chat_template", type=str, default="chatml", help="Chat template for prompt formatting")
    parser.add_argument("--num_fewshot", type=int, default=None, help="Option to subset eval dataset")
    parser.add_argument("--load_in_4bit", type=str, default="False", help="Option to load in 4bit with bitsandbytes")
    parser.add_argument("--query", type=str, default="I need the current stock price of Tesla (TSLA)")
    args = parser.parse_args()

    # specify custom model path
    if args.model_path:
        inference = ModelInference(args.model_path, args.chat_template, args.load_in_4bit)
    else:
        model_path = 'NousResearch/Nous-Hermes-2-PlusPlus'
        inference = ModelInference(model_path, args.chat_template, args.load_in_4bit)
        
    # Run the model evaluator
    inference.generate_function_call(args.query, args.chat_template, args.num_fewshot)
