import datetime
from pydantic import BaseModel
from typing import Dict
from schema import FunctionCall
from utils import (
    get_fewshot_examples
)
import yaml
import json
import os

class PromptSchema(BaseModel):
    Role: str
    Objective: str
    Tools: str
    Examples: str
    Schema: str
    Instructions: str 

class PromptManager:
    def __init__(self):
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        
    def format_yaml_prompt(self, prompt_schema: PromptSchema, variables: Dict) -> str:
        formatted_prompt = ""
        for field, value in prompt_schema.dict().items():
            if field == "Examples" and variables.get("examples") is None:
                continue
            formatted_value = value.format(**variables)
            if field == "Instructions":
                formatted_prompt += f"{formatted_value}"
            else:
                formatted_value = formatted_value.replace("\n", " ")
                formatted_prompt += f"{formatted_value}"
        return formatted_prompt

    def read_yaml_file(self, file_path: str) -> PromptSchema:
        with open(file_path, 'r') as file:
            yaml_content = yaml.safe_load(file)
        
        prompt_schema = PromptSchema(
            Role=yaml_content.get('Role', ''),
            Objective=yaml_content.get('Objective', ''),
            Tools=yaml_content.get('Tools', ''),
            Examples=yaml_content.get('Examples', ''),
            Schema=yaml_content.get('Schema', ''),
            Instructions=yaml_content.get('Instructions', ''),
        )
        return prompt_schema
    
    def generate_prompt(self, user_prompt, tools, num_fewshot=None):
        prompt_path = os.path.join(self.script_dir, 'prompt_assets', 'sys_prompt.yml')
        prompt_schema = self.read_yaml_file(prompt_path)

        if num_fewshot is not None:
            examples = get_fewshot_examples(num_fewshot)
        else:
            examples = None

        schema_json = json.loads(FunctionCall.schema_json())
        #schema = schema_json.get("properties", {})

        variables = {
            "date": datetime.date.today(),
            "tools": tools,
            "examples": examples,
            "schema": schema_json
        }
        sys_prompt = self.format_yaml_prompt(prompt_schema, variables)

        prompt = [
                {'content': sys_prompt, 'role': 'system'}
            ]
        prompt.extend(user_prompt)
        return prompt
        
        
