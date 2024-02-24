import json
from pydantic import ValidationError
from utils import eval_logger
from schema import FunctionCall, FunctionSignature

def validate_function_call_schema(call, signatures):
    try:
        call_data = FunctionCall(**call)
    except ValidationError as e:
        eval_logger.info(f"Invalid function call: {e}")
        return False

    for signature in signatures:
        # Inside the main validation function
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
                                eval_logger.info(f"Invalid argument '{arg_name}': {arg_validation_error}")
                                return False

                # Check if all required arguments are present
                required_arguments = signature_data.function.parameters.get('required', [])
                result, missing_arguments = check_required_arguments(call_data.arguments, required_arguments)

                if not result:
                    eval_logger.info(f"Missing required arguments: {missing_arguments}")
                    return False

                return True
        except Exception as e:
            # Handle validation errors for the function signature
            eval_logger.info(f"Error validating function call: {e}")
            return False

    # Moved the "No matching function signature found" message here
    eval_logger.info(f"No matching function signature found for function: {call_data.name}")
    return False

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