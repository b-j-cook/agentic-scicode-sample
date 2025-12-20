# Background: [Your Step Title]

> **This file is OPTIONAL.** Delete/skip it if the subproblem step doesn't need background context.
> 
> When provided, this content is shown to the LLM when running evaluation with
> `with_background=True`. Use it to provide domain knowledge that helps the model
> understand the problem better.

## What to Include

This is where you provide **domain expertise** that might help solve the problem:

- **Mathematical formulas** the solution should implement
- **Algorithm descriptions** or pseudocode
- **Domain-specific terminology** and definitions
- **Physical intuition** or conceptual explanations
- **References** to papers or resources

## Example Content

### Mathematical Formulation

Use LaTeX for equations:

$$
d = \sqrt{\sum_{i=1}^{n} (x_i - y_i)^2}
$$

Where:
- $d$ is the Euclidean distance
- $x_i$ and $y_i$ are coordinates
- $n$ is the number of dimensions

### Algorithm Notes

Describe any specific approach:

1. First, compute the element-wise differences
2. Square each difference
3. Sum all squared differences
4. Take the square root

### Key Considerations

- Handle edge cases like empty inputs
- Consider numerical stability for large values
- Account for floating-point precision

---

**DELETE EVERYTHING ABOVE** and write your actual background content!
