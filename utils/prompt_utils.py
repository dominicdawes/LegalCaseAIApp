# utils/prompt_utils.py

import os
import yaml
from typing import Dict, Any
from langchain_core.prompts import PromptTemplate


# Where your YAML lives
PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "..", "prompts")


def load_yaml_prompt(file_name: str) -> Dict[str, Any]:
    """
    Given a YAML file (just the file name, e.g. 'chat-persona-prompt.yaml'),
    load it and return a dict with keys: 'llm', 'messages' or 'template', etc.

    Returns:
        - data (Dict): ???
    """
    path = os.path.join(PROMPTS_DIR, file_name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Prompt YAML not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data


def build_chat_messages_from_yaml(yaml_dict: Dict[str, Any]) -> list:
    """
    Given a loaded YAML dict (with a 'messages' key), return a
    list of dicts [ {"role": "...", "content": "..."}, ... ] that you
    can pass into a chat‐style API or stitch into a single string.

    e.g. return

    [
        {"role": "system", "content": "[PERSONA] …"},
        {"role": "system", "content": "[EXAMPLE OUTPUTS] …"},
        {"role": "system", "content": "[IMPORTANT NOTES] …"},
        {"role": "system", "content": "[ESCAPE HATCH] …"},
    ]
    """
    if "messages" not in yaml_dict:
        return []
    return yaml_dict["messages"]  # each item already has {role: ..., content: ...}


def build_prompt_template_from_yaml(yaml_dict: Dict[str, Any]) -> PromptTemplate:
    """
    Given a loaded YAML dict that has a 'template' key, return a
    LangChain PromptTemplate with input variables inserted into the '{}' braces 
    present in the template key.

    E.g. 
    if name=John & template = "Hello {name}!" ---> PromptTemplate="Hello John!"

    See "test/prompt_format_test.py" for example output

    """
    if "template" not in yaml_dict:
        raise KeyError("This YAML does not have a 'template' key")
    template_str: str = yaml_dict["template"]

    # Find all variables in {curly_braces}. A simple heuristic:
    vars = []

    # Very naive: look for {...} patterns.
    for piece in template_str.split("{"):
        if "}" in piece:
            var = piece.split("}")[0].strip()
            if var and var not in vars:
                vars.append(var)

    return PromptTemplate(input_variables=vars, template=template_str)
