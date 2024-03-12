import ast
import os
import re
import json
import logging
import datetime
import xml.etree.ElementTree as ET

from art import text2art
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

def print_nous_text_art(suffix=None):
    font = "nancyj"
    ascii_text = "  nousresearch"
    if suffix:
        ascii_text += f"  x  {suffix}"
    ascii_art = text2art(ascii_text, font=font)
    print(ascii_art)

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
        assistant_pattern = re.compile(r'<\|im_start\|>\s*assistant((?:(?!<\|im_start\|>\s*assistant).)*)$', re.DOTALL)

    elif chat_template == "vicuna":
        assistant_pattern = re.compile(r'ASSISTANT:\s*((?:(?!ASSISTANT:).)*)$', re.DOTALL)
    else:
        raise NotImplementedError(f"Handling for chat_template '{chat_template}' is not implemented.")
    
    assistant_match = assistant_pattern.search(completion)
    if assistant_match:
        assistant_content = assistant_match.group(1).strip()
        if chat_template == "vicuna":
            eos_token = f"</s>{eos_token}"
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
                json_data = None
                try:
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
                except Exception as e:
                    inference_logger.error(f"Cannot strip text: {e}")
                if json_data is not None:
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
