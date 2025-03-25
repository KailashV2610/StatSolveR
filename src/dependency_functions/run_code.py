import subprocess
import re
from src.dependency_functions.functions import *
from src.utils.logger import log_info, log_error, log_progress

def extract_code_block(response: str) -> str:
    """
    Extracts Python code from a response that is enclosed in triple backticks.
    """
    match = re.search(r"```(?:python)?\n(.*?)```", response, re.DOTALL)
    log_info(f"Extracted code block: {match.group(1)}")
    return match.group(1).strip() if match else None

def run_code(code: str) -> tuple:
    """Runs the provided Python code and returns the output."""
    code = extract_code_block(code)
    try:
        result = subprocess.run(["python3", "-c", code], capture_output=True, text=True)
        log_info(f"Code execution result: {result.stdout if result.stdout else result.stderr}")
        return code, result.stdout if result.stdout else result.stderr
    except Exception as e:
        return str(e)