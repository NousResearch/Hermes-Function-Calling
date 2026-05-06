import argparse
import json

FLASH_ATTENTION_TROUBLESHOOTING = (
    "Failed to load the model with FlashAttention 2. This usually means the installed "
    "flash-attn wheel is not ABI-compatible with your PyTorch/CUDA build. Reinstall "
    "flash-attn for the active environment, for example `pip uninstall flash-attn && "
    "pip install --no-build-isolation flash-attn`, or run with `--attn_implementation "
    "eager` to disable FlashAttention."
)


def is_flash_attention_import_error(error):
    message = str(error).lower()
    return (
        "flash_attn" in message
        or "flash-attn" in message
        or "flash attention" in message
    )


def load_runtime_dependencies():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    import functions
    from prompter import PromptManager
    from validator import validate_function_call_schema
    from utils import (
        print_nous_text_art,
        inference_logger,
        get_assistant_message,
        get_chat_template,
        validate_and_extract_tool_calls,
    )

    return {
        "torch": torch,
        "AutoModelForCausalLM": AutoModelForCausalLM,
        "AutoTokenizer": AutoTokenizer,
        "BitsAndBytesConfig": BitsAndBytesConfig,
        "functions": functions,
        "PromptManager": PromptManager,
        "validate_function_call_schema": validate_function_call_schema,
        "print_nous_text_art": print_nous_text_art,
        "inference_logger": inference_logger,
        "get_assistant_message": get_assistant_message,
        "get_chat_template": get_chat_template,
        "validate_and_extract_tool_calls": validate_and_extract_tool_calls,
    }


class ModelInference:
    def __init__(self, model_path, chat_template, load_in_4bit, attn_implementation):
        deps = load_runtime_dependencies()
        self.functions = deps["functions"]
        self.validate_function_call_schema = deps["validate_function_call_schema"]
        self.inference_logger = deps["inference_logger"]
        self.get_assistant_message = deps["get_assistant_message"]
        self.get_chat_template = deps["get_chat_template"]
        self.validate_and_extract_tool_calls = deps["validate_and_extract_tool_calls"]

        self.inference_logger.info(deps["print_nous_text_art"]())
        self.prompter = deps["PromptManager"]()
        self.bnb_config = None

        if load_in_4bit == "True":
            self.bnb_config = deps["BitsAndBytesConfig"](
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        try:
            self.model = deps["AutoModelForCausalLM"].from_pretrained(
                model_path,
                trust_remote_code=True,
                return_dict=True,
                quantization_config=self.bnb_config,
                torch_dtype=deps["torch"].float16,
                attn_implementation=attn_implementation,
                device_map="auto",
            )
        except (ImportError, OSError, RuntimeError) as exc:
            if (
                attn_implementation == "flash_attention_2"
                and is_flash_attention_import_error(exc)
            ):
                raise RuntimeError(FLASH_ATTENTION_TROUBLESHOOTING) from exc
            raise

        self.tokenizer = deps["AutoTokenizer"].from_pretrained(
            model_path, trust_remote_code=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        if self.tokenizer.chat_template is None:
            print("No chat template defined, getting chat_template...")
            self.tokenizer.chat_template = self.get_chat_template(chat_template)

        self.inference_logger.info(self.model.config)
        self.inference_logger.info(self.model.generation_config)
        self.inference_logger.info(self.tokenizer.special_tokens_map)

    def process_completion_and_validate(self, completion, chat_template):

        assistant_message = self.get_assistant_message(
            completion, chat_template, self.tokenizer.eos_token
        )

        if assistant_message:
            validation, tool_calls, error_message = (
                self.validate_and_extract_tool_calls(assistant_message)
            )

            if validation:
                self.inference_logger.info(
                    f"parsed tool calls:\n{json.dumps(tool_calls, indent=2)}"
                )
                return tool_calls, assistant_message, error_message
            else:
                tool_calls = None
                return tool_calls, assistant_message, error_message
        else:
            self.inference_logger.warning("Assistant message is None")
            raise ValueError("Assistant message is None")

    def execute_function_call(self, tool_call):
        function_name = tool_call.get("name")
        function_to_call = getattr(self.functions, function_name, None)
        function_args = tool_call.get("arguments", {})

        self.inference_logger.info(f"Invoking function call {function_name} ...")
        function_response = function_to_call(*function_args.values())
        results_dict = f'{{"name": "{function_name}", "content": {function_response}}}'
        return results_dict

    def run_inference(self, prompt):
        inputs = self.tokenizer.apply_chat_template(
            prompt, add_generation_prompt=True, return_tensors="pt"
        )

        tokens = self.model.generate(
            inputs.to(self.model.device),
            max_new_tokens=1500,
            temperature=0.8,
            repetition_penalty=1.1,
            do_sample=True,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        completion = self.tokenizer.decode(
            tokens[0], skip_special_tokens=False, clean_up_tokenization_space=True
        )
        return completion

    def generate_function_call(self, query, chat_template, num_fewshot, max_depth=5):
        try:
            depth = 0
            user_message = f"{query}\nThis is the first turn and you don't have <tool_results> to analyze yet"
            chat = [{"role": "user", "content": user_message}]
            tools = self.functions.get_openai_tools()
            prompt = self.prompter.generate_prompt(chat, tools, num_fewshot)
            completion = self.run_inference(prompt)

            def recursive_loop(prompt, completion, depth):
                nonlocal max_depth
                tool_calls, assistant_message, error_message = (
                    self.process_completion_and_validate(completion, chat_template)
                )
                prompt.append({"role": "assistant", "content": assistant_message})

                tool_message = (
                    f"Agent iteration {depth} to assist with user query: {query}\n"
                )
                if tool_calls:
                    self.inference_logger.info(
                        f"Assistant Message:\n{assistant_message}"
                    )

                    for tool_call in tool_calls:
                        validation, message = self.validate_function_call_schema(
                            tool_call, tools
                        )
                        if validation:
                            try:
                                function_response = self.execute_function_call(
                                    tool_call
                                )
                                tool_message += f"<tool_response>\n{function_response}\n</tool_response>\n"
                                self.inference_logger.info(
                                    f"Here's the response from the function call: {tool_call.get('name')}\n{function_response}"
                                )
                            except Exception as e:
                                self.inference_logger.info(
                                    f"Could not execute function: {e}"
                                )
                                tool_message += f"<tool_response>\nThere was an error when executing the function: {tool_call.get('name')}\nHere's the error traceback: {e}\nPlease call this function again with correct arguments within XML tags <tool_call></tool_call>\n</tool_response>\n"
                        else:
                            self.inference_logger.info(message)
                            tool_message += f"<tool_response>\nThere was an error validating function call against function signature: {tool_call.get('name')}\nHere's the error traceback: {message}\nPlease call this function again with correct arguments within XML tags <tool_call></tool_call>\n</tool_response>\n"
                    prompt.append({"role": "tool", "content": tool_message})

                    depth += 1
                    if depth >= max_depth:
                        print(
                            f"Maximum recursion depth reached ({max_depth}). Stopping recursion."
                        )
                        return

                    completion = self.run_inference(prompt)
                    recursive_loop(prompt, completion, depth)
                elif error_message:
                    self.inference_logger.info(
                        f"Assistant Message:\n{assistant_message}"
                    )
                    tool_message += f"<tool_response>\nThere was an error parsing function calls\n Here's the error stack trace: {error_message}\nPlease call the function again with correct syntax<tool_response>"
                    prompt.append({"role": "tool", "content": tool_message})

                    depth += 1
                    if depth >= max_depth:
                        print(
                            f"Maximum recursion depth reached ({max_depth}). Stopping recursion."
                        )
                        return

                    completion = self.run_inference(prompt)
                    recursive_loop(prompt, completion, depth)
                else:
                    self.inference_logger.info(
                        f"Assistant Message:\n{assistant_message}"
                    )

            recursive_loop(prompt, completion, depth)

        except Exception as e:
            self.inference_logger.error(f"Exception occurred: {e}")
            raise e


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run recursive function calling loop")
    parser.add_argument("--model_path", type=str, help="Path to the model folder")
    parser.add_argument(
        "--chat_template",
        type=str,
        default="chatml",
        help="Chat template for prompt formatting",
    )
    parser.add_argument(
        "--num_fewshot", type=int, default=None, help="Option to use json mode examples"
    )
    parser.add_argument(
        "--load_in_4bit",
        type=str,
        default="False",
        help="Option to load in 4bit with bitsandbytes",
    )
    parser.add_argument(
        "--query", type=str, default="I need the current stock price of Tesla (TSLA)"
    )
    parser.add_argument(
        "--max_depth", type=int, default=5, help="Maximum number of recursive iteration"
    )
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="flash_attention_2",
        choices=["flash_attention_2", "eager", "sdpa"],
        help="Attention implementation passed to transformers.from_pretrained",
    )
    args = parser.parse_args()

    # specify custom model path
    if args.model_path:
        inference = ModelInference(
            args.model_path,
            args.chat_template,
            args.load_in_4bit,
            args.attn_implementation,
        )
    else:
        model_path = "NousResearch/Hermes-2-Pro-Llama-3-8B"
        inference = ModelInference(
            model_path, args.chat_template, args.load_in_4bit, args.attn_implementation
        )

    # Run the model evaluator
    inference.generate_function_call(
        args.query, args.chat_template, args.num_fewshot, args.max_depth
    )
