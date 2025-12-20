"""
WRITE YOUR STEP DESCRIPTION HERE (this entire docstring is shown to the LLM)

This is the "step_description_prompt" - the instructions the model receives.
Write it as if you're explaining the task to a skilled programmer who needs
to implement this function from scratch.

Include:
- Clear description of what the function should do
- Any mathematical formulas or algorithms to use
- Edge cases to handle
- Constraints (e.g., "must handle negative values")

Example of a good description:
    "Implement a function to calculate the Euclidean distance between two points
    in N-dimensional space. The function should handle arrays of any dimension
    and return a scalar distance value."

DELETE THIS COMMENT BLOCK when writing your actual description.
"""

import numpy as np
# Add any imports your gold solution needs


# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================
# The LLM sees:
#   1. This function's name and parameters
#   2. The docstring (so write a clear one!)
#   3. A "return hint" extracted from the return statement below
#
# The "return hint" tells the model what to return. For example:
#   return result           →  hint: "return result"
#   return x, y, z          →  hint: "return x, y, z"  
#   return {"a": a, "b": b} →  hint: "return {"a": a, "b": b}"
#
# This helps the model know the expected output structure.

def function_name(param1, param2):
    '''Docstring shown to the LLM - make it clear and helpful.
    
    Parameters
    ----------
    param1 : type
        Description of param1.
    param2 : type
        Description of param2.
    
    Returns
    -------
    result : type
        Description of what is returned.
    '''
    return result  # ← This line becomes the "return hint" shown to the model


# =============================================================================
# GOLD SOLUTION (your reference implementation - NOT shown to LLM)
# =============================================================================
# This is YOUR correct implementation. It is used to:
# 1. Automatically generate expected test outputs
# 2. Verify the task is solvable
# 3. Provide ground truth for scoring
#
# NAMING RULE: Must be named _gold_{function_name}
# Example: if your function is "calculate_distance", gold is "_gold_calculate_distance"

def _gold_function_name(param1, param2):
    '''Your correct implementation goes here.'''
    # IMPLEMENT YOUR SOLUTION HERE
    # This code generates the expected outputs for test cases
    result = None  # Replace with your actual implementation
    return result


# =============================================================================
# TEST CASES
# =============================================================================
# Define test inputs. The compiler will:
# 1. Run "setup" to create variables
# 2. Run "gold_call" to get the expected output
# 3. Store expected output for evaluation
#
# During evaluation, the LLM's code is tested with:
#   setup code + assert(model_output == expected_output)

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            # Code that sets up test variables
            "setup": """
param1 = ...  # Your test input
param2 = ...  # Your test input
""",
            # How to call the function being tested
            "call": "function_name(param1, param2)",
            # How to call your gold solution (same args)
            "gold_call": "_gold_function_name(param1, param2)",
        },
        {
            "setup": """
param1 = ...  # Different test input
param2 = ...  # Different test input
""",
            "call": "function_name(param1, param2)",
            "gold_call": "_gold_function_name(param1, param2)",
        },
        # Add 3+ test cases covering:
        # - Normal/typical inputs
        # - Edge cases (empty arrays, zero values, etc.)
        # - Boundary conditions
    ]
