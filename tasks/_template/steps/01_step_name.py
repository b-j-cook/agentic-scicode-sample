"""
TODO: Step Title

TODO: Describe what this step implements. This text becomes the 
step_description_prompt shown to the LLM.

Be specific about:
- What function to implement
- What it should do
- Any constraints or edge cases
"""

import numpy as np
# TODO: Add other imports your gold solution needs


# =============================================================================
# FUNCTION SIGNATURE
# =============================================================================
# This is what the LLM sees and must implement.
# 
# IMPORTANT:
# - The function body is stripped out during compilation
# - Only the signature and docstring are shown to the model
# - The return statement becomes the "return_line" hint

def function_name(param1, param2):
    '''TODO: Write a clear docstring.
    
    Parameters:
    -----------
    param1 : type
        Description of param1.
    param2 : type
        Description of param2.
    
    Returns:
    --------
    result : type
        Description of what is returned.
    
    Notes:
    ------
    [Optional: Add any helpful notes about the implementation]
    
    Examples:
    ---------
    [Optional: Add usage examples]
    >>> function_name(1, 2)
    3
    '''
    return result


# =============================================================================
# GOLD SOLUTION
# =============================================================================
# The reference implementation that:
# 1. Generates test targets automatically
# 2. Validates test cases work correctly  
# 3. Provides ground truth for evaluation
#
# NAMING: Must be _gold_{function_name}

def _gold_function_name(param1, param2):
    '''Reference implementation.'''
    # TODO: Implement the correct solution
    result = None  # Replace with actual implementation
    return result


# =============================================================================
# TEST CASES
# =============================================================================
# Each test case specifies:
#   - setup: Code to create test inputs
#   - call: How to call the function being tested  
#   - gold_call: How to call the gold solution
#
# The compiler will:
# 1. Run setup code
# 2. Execute gold_call to get expected output
# 3. Store output in H5 file as "target"
# 4. Generate test: setup + "assert comparison(call, target)"

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": """
# TODO: Set up test inputs
param1 = None
param2 = None
""",
            "call": "function_name(param1, param2)",
            "gold_call": "_gold_function_name(param1, param2)",
        },
        {
            "setup": """
# TODO: Set up different test inputs
param1 = None
param2 = None
""",
            "call": "function_name(param1, param2)",
            "gold_call": "_gold_function_name(param1, param2)",
        },
        # Add more test cases as needed
        # Aim for 3+ test cases covering:
        # - Normal/typical inputs
        # - Edge cases  
        # - Boundary conditions
    ]
