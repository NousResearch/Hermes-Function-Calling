import ast
import os
import re
import json
import logging
import datetime
import xml.etree.ElementTree as ET
from logging.handlers import RotatingFileHandler

logging.basicConfig(
    format="%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
    level=logging.INFO,
)
script_dir = os.path.dirname(os.path.abspath(__file__))
now = datetime.datetime.now()
log_folder = os.path.join(script_dir, "inference_logs")
os.makedirs(log_folder, exist_ok=True)
log_file_path = os.path.join(
    log_folder, f"function-calling-inference_{now.strftime('%Y-%m-%d_%H-%M-%S')}.log"
)
# Use RotatingFileHandler from the logging.handlers module
file_handler = RotatingFileHandler(log_file_path, maxBytes=0, backupCount=0)
file_handler.setLevel(logging.INFO)

formatter = logging.Formatter("%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s", datefmt="%Y-%m-%d:%H:%M:%S")
file_handler.setFormatter(formatter)

inference_logger = logging.getLogger("function-calling-inference")
inference_logger.addHandler(file_handler)

def get_fewshot_examples(num_fewshot):
    """return a list of few shot examples"""
    example_path = os.path.join(script_dir, 'prompt_assets', 'few_shot.json')
    with open(example_path, 'r') as file:
        examples = json.load(file)  # Use json.load with the file object, not the file path
    if num_fewshot > len(examples):
        raise ValueError(f"Not enough examples (got {num_fewshot}, but there are only {len(examples)} examples).")
    return examples[:num_fewshot]

def get_chat_template(chat_template):
    """read chat template from jinja file"""
    template_path = os.path.join(script_dir, 'chat_templates', f"{chat_template}.j2")

    if not os.path.exists(template_path):
        print
        inference_logger.error(f"Template file not found: {chat_template}")
        return None
    try:
        with open(template_path, 'r') as file:
            template = file.read()
        return template
    except Exception as e:
        print(f"Error loading template: {e}")
        return None

def get_assistant_message(completion, chat_template, eos_token):
    """define and match pattern to find the assistant message"""
    completion = completion.strip()

    if chat_template == "zephyr":
        assistant_pattern = re.compile(r'<\|assistant\|>((?:(?!<\|assistant\|>).)*)$', re.DOTALL)
    elif chat_template == "chatml":
        assistant_pattern = re.compile(r'<\\|im_start\\|>\s*assistant((?:(?!<\\|im_start\\|>\s*assistant).)*)$', re.DOTALL)
    else:
        raise NotImplementedError(f"Handling for chat_template '{chat_template}' is not implemented.")
    
    assistant_match = assistant_pattern.search(completion)
    if assistant_match:
        assistant_content = assistant_match.group(1).strip()
        return assistant_content.replace(eos_token, "")
    else:
        assistant_content = None
        inference_logger.info("No match found for the assistant pattern")
        return assistant_content

def validate_and_extract_tool_calls(assistant_content):
        validation_result = False
        tool_calls = []
        try:
            # wrap content in root element
            xml_root_element = f"<root>{assistant_content}</root>"
            root = ET.fromstring(xml_root_element)

            # extract JSON data
            for element in root.findall(".//tool_call"):
                json_text = element.text.strip()

                try:
                # Prioritize json.loads for better error handling
                    json_data = json.loads(json_text)
                except json.JSONDecodeError as json_err:
                    try:
                        # Fallback to ast.literal_eval if json.loads fails
                        json_data = ast.literal_eval(json_text)
                    except (SyntaxError, ValueError) as eval_err:
                        inference_logger.error("JSON parsing failed with both json.loads and ast.literal_eval:")
                        inference_logger.error("- JSON Decode Error: %s", json_err)
                        inference_logger.error("- Fallback Syntax/Value Error: %s", eval_err)
                        inference_logger.error("- Problematic JSON text: %s", json_text)
                        continue

                tool_calls.append(json_data)
                validation_result = True

        except ET.ParseError as err:
            inference_logger.error("XML Parse Error: %s", err)


        # Return default values if no valid data is extracted
        return validation_result, tool_calls

def validate_tool_calls(generated_arguments, expected_arguments):
    for key, expected_value in expected_arguments.items():
        if generated_arguments.get(key) != expected_value:
            inference_logger.info("Expected: %s", expected_value)
            inference_logger.info("Got: %s", generated_arguments.get(key))
            return "failed"
    return "passed"

def calculate_pass_rate(eval_results):
    passed_count =sum(1 for sample in eval_results if sample["result"] == "passed")
    inference_logger.info("Number of eval tests passed: %s", passed_count)
    inference_logger.info("Number of eval tests failed: %s", len(eval_results) - passed_count)

    pass_rate = passed_count / len(eval_results)
    inference_logger.info(f"fireworks-ai function-calling eval (pass@1): {pass_rate}")
    return pass_rate