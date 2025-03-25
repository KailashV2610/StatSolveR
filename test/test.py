import pytest
from unittest.mock import patch, MagicMock
from src.dependency_functions.functions import retrieve_context, ask_llm
from src.dependency_functions.run_code import run_code

# Sample input query
sample_query = "Calculate the mean and standard deviation for the given dataset: [10, 20, 30, 40, 50]"

# Mock responses
mock_generated_code = """
```python
import numpy as np
data = [10, 20, 30, 40, 50]
mean = np.mean(data)
std_dev = np.std(data)
print(f"Mean: {mean}, Standard Deviation: {std_dev}")
```
"""

@patch("src.dependency_functions.functions.retrieve_context", return_value=mock_generated_code)
def test_generate_code(mock_gpt):
    """ Test if GPT-4 generates valid Python code """
    response = ask_llm(sample_query)
    assert "import numpy as np" in response
    assert "np.mean(data)" in response
    assert "np.std(data)" in response

@patch("src.dependency_functions.run_code")
def test_execute_code(mock_exec):
    """ Test if the generated code executes without errors """
    try:
        run_code(mock_generated_code)
        assert True
    except Exception:
        assert False, "Generated code execution failed!"

if __name__ == "__main__":
    pytest.main()
