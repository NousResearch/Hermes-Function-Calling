import ast
import json
from jsonschema import validate
from pydantic import ValidationError
from utils import inference_logger, extract_json_from_markdown
from schema import FunctionCall, FunctionSignature

def validate_function_call_schema(call, signatures):
    try:
        call_data = FunctionCall(**call)
    except ValidationError as e:
        return False, str(e)

    for signature in signatures:
        try:
            signature_data = FunctionSignature(**signature)
            if signature_data.function.name == call_data.name:
                # Validate types in function arguments
                for arg_name, arg_schema in signature_data.function.parameters.get('properties', {}).items():
                    if arg_name in call_data.arguments:
                        call_arg_value = call_data.arguments[arg_name]
                        if call_arg_value:
                            try:
                                validate_argument_type(arg_name, call_arg_value, arg_schema)
                            except Exception as arg_validation_error:
                                return False, str(arg_validation_error)

                # Check if all required arguments are present
                required_arguments = signature_data.function.parameters.get('required', [])
                result, missing_arguments = check_required_arguments(call_data.arguments, required_arguments)
                if not result:
                    return False, f"Missing required arguments: {missing_arguments}"

                return True, None
        except Exception as e:
            # Handle validation errors for the function signature
            return False, str(e)

    # No matching function signature found
    return False, f"No matching function signature found for function: {call_data.name}"

def check_required_arguments(call_arguments, required_arguments):
    missing_arguments = [arg for arg in required_arguments if arg not in call_arguments]
    return not bool(missing_arguments), missing_arguments

def validate_enum_value(arg_name, arg_value, enum_values):
    if arg_value not in enum_values:
        raise Exception(
            f"Invalid value '{arg_value}' for parameter {arg_name}. Expected one of {', '.join(map(str, enum_values))}"
        )

def validate_argument_type(arg_name, arg_value, arg_schema):
    arg_type = arg_schema.get('type', None)
    if arg_type:
        if arg_type == 'string' and 'enum' in arg_schema:
            enum_values = arg_schema['enum']
            if None not in enum_values and enum_values != []:
                try:
                    validate_enum_value(arg_name, arg_value, enum_values)
                except Exception as e:
                    # Propagate the validation error message
                    raise Exception(f"Error validating function call: {e}")

        python_type = get_python_type(arg_type)
        if not isinstance(arg_value, python_type):
            raise Exception(f"Type mismatch for parameter {arg_name}. Expected: {arg_type}, Got: {type(arg_value)}")

def get_python_type(json_type):
    type_mapping = {
        'string': str,
        'number': (int, float),
        'integer': int,
        'boolean': bool,
        'array': list,
        'object': dict,
        'null': type(None),
    }
    return type_mapping[json_type]

def validate_json_data(json_object, json_schema):
    valid = False
    error_message = None
    result_json = None

    try:
        # Attempt to load JSON using json.loads
        try:
            result_json = json.loads(json_object)
        except json.decoder.JSONDecodeError:
            # If json.loads fails, try ast.literal_eval
            try:
                result_json = ast.literal_eval(json_object)
            except (SyntaxError, ValueError) as e:
                try:
                    result_json = extract_json_from_markdown(json_object)
                except Exception as e:
                    error_message = f"JSON decoding error: {e}"
                    inference_logger.info(f"Validation failed for JSON data: {error_message}")
                    return valid, result_json, error_message

        # Return early if both json.loads and ast.literal_eval fail
        if result_json is None:
            error_message = "Failed to decode JSON data"
            inference_logger.info(f"Validation failed for JSON data: {error_message}")
            return valid, result_json, error_message

        # Validate each item in the list against schema if it's a list
        if isinstance(result_json, list):
            for index, item in enumerate(result_json):
                try:
                    validate(instance=item, schema=json_schema)
                    inference_logger.info(f"Item {index+1} is valid against the schema.")
                except ValidationError as e:
                    error_message = f"Validation failed for item {index+1}: {e}"
                    break
        else:
            # Default to validation without list
            try:
                validate(instance=result_json, schema=json_schema)
            except ValidationError as e:
                error_message = f"Validation failed: {e}"

    except Exception as e:
        error_message = f"Error occurred: {e}"

    if error_message is None:
        valid = True
        inference_logger.info("JSON data is valid against the schema.")
    else:
        inference_logger.info(f"Validation failed for JSON data: {error_message}")

    return valid, result_json, error_message