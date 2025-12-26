#!/usr/bin/env python3
"""
Paper Screening QC for SciCode Benchmark

Evaluates whether a research paper is suitable for conversion into a SciCode task.
Runs BEFORE any task authoring begins.

Usage:
    python screen_paper.py --pdf path/to/paper.pdf
    python screen_paper.py --pdf "paper1.pdf paper2.pdf"
    python screen_paper.py --pdf candidates/
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

try:
    import fitz  # pymupdf
except ImportError:
    print("Please install pymupdf: pip install pymupdf")
    sys.exit(1)

try:
    from openai import OpenAI
except ImportError:
    print("Please install openai: pip install openai")
    sys.exit(1)


# =============================================================================
# PDF EXTRACTION
# =============================================================================

def extract_pdf_text(pdf_path: Path, max_pages: int = 25) -> str:
    """
    Extract text content from a PDF file.
    
    Args:
        pdf_path: Path to PDF file
        max_pages: Maximum pages to extract (to manage context length)
    
    Returns:
        Extracted text with page markers
    """
    doc = fitz.open(pdf_path)
    text_parts = []
    
    for i, page in enumerate(doc):
        if i >= max_pages:
            remaining = len(doc) - max_pages
            text_parts.append(f"\n\n[... {remaining} additional pages not shown ...]")
            break
        text_parts.append(f"=== PAGE {i + 1} ===\n{page.get_text()}")
    
    doc.close()
    return "\n\n".join(text_parts)


def extract_pdf_metadata(pdf_path: Path) -> dict:
    """Extract metadata from PDF (title, authors, etc.)."""
    doc = fitz.open(pdf_path)
    metadata = dict(doc.metadata) if doc.metadata else {}
    metadata["page_count"] = len(doc)
    doc.close()
    return metadata


# =============================================================================
# SCREENING PROMPT
# =============================================================================

SCREENING_PROMPT = '''You are helping curate tasks for the SciCode benchmark, which evaluates language models on realistic scientific programming problems across physics, math, materials science, biology, and chemistry. SciCode tasks are derived from real research problems, focus on numerical methods, simulation of systems, and scientific calculation, and are intended to be challenging even for current frontier models.

You will be given a PDF describing a scientific research problem, method, or study. Your job is to decide whether this PDF is a good candidate for conversion into a SciCode-style coding task.

## THE PAPER

{pdf_content}

---

## EVALUATION CRITERIA

A good candidate should satisfy ALL of the following:

### 1. CONCRETE COMPUTATIONAL PROBLEM
Describes a concrete computational or numerical problem that can realistically be implemented in code (preferably Python) and tested with deterministic or domain-specific test cases.

### 2. NON-TRIVIAL COMPLEXITY
Solving it should require multiple steps of scientific reasoning and coding (e.g., designing or applying numerical methods, simulations, or scientific calculations), not just a short formula plug-in or textbook exercise.

### 3. CORE ALGORITHM SPECIFIED
The PDF should describe the core algorithm or method clearly enough that its key steps can be implemented. It does NOT need to be 100% self-contained — task authors will add background context and fill in standard details. Focus on whether the paper specifies:
- The main algorithmic steps or mathematical procedure
- Key equations or formulas
- What makes this method distinct

External datasets, evaluation benchmarks, or missing implementation details are NOT blockers — only reject if the core method itself is too vague to implement.

---

## BLIND IMPLEMENTATION TEST (CRITICAL)

Before carefully analyzing the paper, attempt to implement the core algorithm/method from your existing knowledge alone. This tests whether you already know how to solve this.

<blind_implementation_attempt>
Based only on the title and general topic, write pseudocode or Python code for the core algorithm. Be honest - write what you actually know, not what you're inferring from the paper.

If you find yourself unable to write meaningful code without reading the paper details, state that clearly.
</blind_implementation_attempt>

---

## YOUR EVALUATION

Provide your assessment in the following JSON format:

```json
{{
  "paper_info": {{
    "title": "extracted or inferred title",
    "domain": "physics|math|materials|biology|chemistry|other",
    "summary": "2-3 sentence summary of what the paper describes"
  }},
  
  "blind_implementation": {{
    "attempted": true,
    "could_implement": true|false,
    "code_or_pseudocode": "what you wrote in the blind attempt",
    "confidence": 0.0-1.0,
    "explanation": "why you could or couldn't implement this"
  }},
  
  "criteria": {{
    "concrete_computational": {{
      "satisfied": true|false,
      "score": 1-5,
      "reasoning": "..."
    }},
    "nontrivial_complexity": {{
      "satisfied": true|false,
      "score": 1-5,
      "estimated_steps": 3-8,
      "reasoning": "..."
    }},
    "core_algorithm_specified": {{
      "satisfied": true|false,
      "score": 1-5,
      "has_equations": true|false,
      "has_algorithm": true|false,
      "reasoning": "..."
    }}
  }},
  
  "red_flags": [
    "list any concerns (requires special hardware, proprietary data, etc.)"
  ],
  
  "overall_verdict": "APPROVE|REJECT|NEEDS_REVIEW",
  "confidence": 0.0-1.0,
  "reasoning": "2-3 sentence justification for the verdict",
  
  "if_approved": {{
    "suggested_task_name": "snake_case_name",
    "suggested_domain": "physics|math|materials|biology|chemistry",
    "suggested_steps": ["step 1 description", "step 2 description", "..."],
    "implementation_notes": "any helpful notes for the task author"
  }}
}}
```

## DECISION RULES

- **REJECT** if: blind implementation succeeded with high confidence (>0.7), OR any criterion scores 1-2, OR has major red flags
- **NEEDS_REVIEW** if: blind implementation partially succeeded (confidence 0.4-0.7), OR missing some details but recoverable
- **APPROVE** if: all criteria score 3+, blind implementation failed or was very incomplete (confidence <0.4), no major red flags

The blind implementation test is the primary signal for novelty. If you cannot implement the core algorithm without reading the paper, it's likely novel enough.
'''


# =============================================================================
# SCREENING LOGIC
# =============================================================================

def screen_paper(pdf_path: Path, client: OpenAI, model: str) -> dict:
    """
    Screen a single paper for SciCode suitability.
    
    Args:
        pdf_path: Path to PDF file
        client: OpenAI client
        model: Model to use for screening
    
    Returns:
        Screening result dictionary
    """
    print(f"  Extracting text from {pdf_path.name}...")
    pdf_text = extract_pdf_text(pdf_path)
    metadata = extract_pdf_metadata(pdf_path)
    
    # Truncate if too long
    max_chars = 80000
    if len(pdf_text) > max_chars:
        pdf_text = pdf_text[:max_chars] + "\n\n[... content truncated for length ...]"
    
    print(f"  Extracted {len(pdf_text):,} chars from {metadata.get('page_count', '?')} pages")
    print(f"  Sending to {model} for evaluation...")
    
    prompt = SCREENING_PROMPT.format(pdf_content=pdf_text)
    
    response = client.chat.completions.create(
        model=model,
        max_completion_tokens=4096,
        messages=[{"role": "user", "content": prompt}]
    )
    
    response_text = response.choices[0].message.content
    
    # Parse JSON from response
    try:
        # Find JSON block
        start = response_text.find("{")
        end = response_text.rfind("}") + 1
        if start >= 0 and end > start:
            result = json.loads(response_text[start:end])
        else:
            raise ValueError("No JSON found in response")
    except (json.JSONDecodeError, ValueError) as e:
        result = {
            "error": f"Failed to parse response: {e}",
            "raw_response": response_text[:2000]
        }
    
    # Add metadata
    result["_meta"] = {
        "pdf_file": pdf_path.name,
        "pdf_path": str(pdf_path),
        "pdf_pages": metadata.get("page_count"),
        "model_used": model
    }
    
    return result


# =============================================================================
# REPORT GENERATION
# =============================================================================

def generate_markdown_report(results: list[dict]) -> str:
    """Generate a markdown report from screening results."""
    lines = [
        "# 📄 Paper Screening Report",
        "",
        "> Automated screening for SciCode benchmark suitability",
        ""
    ]
    
    # Summary table
    lines.append("## Summary")
    lines.append("")
    lines.append("| Paper | Verdict | Complexity | Algorithm Clear | Blind Test |")
    lines.append("|-------|---------|------------|-----------------|------------|")
    
    for r in results:
        if r.get("error"):
            lines.append(f"| {r['_meta']['pdf_file']} | ❌ ERROR | - | - | - |")
            continue
        
        verdict = r.get("overall_verdict", "?")
        emoji = {"APPROVE": "✅", "REJECT": "❌", "NEEDS_REVIEW": "⚠️"}.get(verdict, "❓")
        complexity = r.get("criteria", {}).get("nontrivial_complexity", {}).get("score", "?")
        algo_clear = r.get("criteria", {}).get("core_algorithm_specified", {}).get("score", "?")
        blind_conf = r.get("blind_implementation", {}).get("confidence", "?")
        if isinstance(blind_conf, float):
            blind_conf = f"{blind_conf:.0%}"
        
        title = r.get("paper_info", {}).get("title", r["_meta"]["pdf_file"])
        if len(title) > 50:
            title = title[:47] + "..."
        
        lines.append(f"| {title} | {emoji} {verdict} | {complexity}/5 | {algo_clear}/5 | {blind_conf} |")
    
    lines.append("")
    
    # Detailed results
    lines.append("## Detailed Results")
    lines.append("")
    
    for r in results:
        meta = r.get("_meta", {})
        pdf_file = meta.get("pdf_file", "unknown")
        
        if r.get("error"):
            lines.append(f"### ❌ {pdf_file}")
            lines.append("")
            lines.append(f"**Error:** {r['error']}")
            lines.append("")
            if r.get("raw_response"):
                lines.append("<details>")
                lines.append("<summary>Raw response</summary>")
                lines.append("")
                lines.append("```")
                lines.append(r["raw_response"])
                lines.append("```")
                lines.append("</details>")
            lines.append("")
            continue
        
        # Header
        verdict = r.get("overall_verdict", "UNKNOWN")
        emoji = {"APPROVE": "✅", "REJECT": "❌", "NEEDS_REVIEW": "⚠️"}.get(verdict, "❓")
        paper_info = r.get("paper_info", {})
        title = paper_info.get("title", pdf_file)
        
        lines.append(f"### {emoji} {title}")
        lines.append("")
        lines.append(f"**File:** `{pdf_file}`")
        lines.append(f"**Domain:** {paper_info.get('domain', 'unknown')}")
        lines.append(f"**Verdict:** {verdict} (confidence: {r.get('confidence', '?')})")
        lines.append("")
        lines.append(f"**Summary:** {paper_info.get('summary', 'N/A')}")
        lines.append("")
        
        # Blind implementation (the key test)
        blind = r.get("blind_implementation", {})
        lines.append("#### 🧪 Blind Implementation Test")
        lines.append("")
        could_impl = blind.get("could_implement", False)
        impl_emoji = "⚠️ YES (concerning)" if could_impl else "✅ NO (good)"
        lines.append(f"- **Could implement from memory:** {impl_emoji}")
        lines.append(f"- **Confidence:** {blind.get('confidence', '?')}")
        lines.append(f"- **Explanation:** {blind.get('explanation', 'N/A')}")
        lines.append("")
        
        if blind.get("code_or_pseudocode"):
            lines.append("<details>")
            lines.append("<summary>Blind attempt code</summary>")
            lines.append("")
            lines.append("```python")
            lines.append(blind["code_or_pseudocode"])
            lines.append("```")
            lines.append("</details>")
            lines.append("")
        
        # Criteria scores
        criteria = r.get("criteria", {})
        lines.append("#### 📊 Criteria Scores")
        lines.append("")
        
        for crit_name, crit_key in [
            ("Concrete Computational", "concrete_computational"),
            ("Non-trivial Complexity", "nontrivial_complexity"),
            ("Core Algorithm Specified", "core_algorithm_specified")
        ]:
            crit = criteria.get(crit_key, {})
            score = crit.get("score", "?")
            satisfied = "✓" if crit.get("satisfied") else "✗"
            lines.append(f"- **{crit_name}:** {score}/5 {satisfied}")
            if crit.get("reasoning"):
                lines.append(f"  - {crit['reasoning']}")
        
        lines.append("")
        
        # Red flags
        red_flags = r.get("red_flags", [])
        if red_flags:
            lines.append("#### 🚩 Red Flags")
            lines.append("")
            for flag in red_flags:
                lines.append(f"- {flag}")
            lines.append("")
        
        # If approved, show suggestions
        if verdict == "APPROVE" and r.get("if_approved"):
            approved = r["if_approved"]
            lines.append("#### 💡 Implementation Suggestions")
            lines.append("")
            lines.append(f"- **Suggested task name:** `{approved.get('suggested_task_name', 'N/A')}`")
            lines.append(f"- **Domain:** {approved.get('suggested_domain', 'N/A')}")
            if approved.get("suggested_steps"):
                lines.append("- **Suggested steps:**")
                for i, step in enumerate(approved["suggested_steps"], 1):
                    lines.append(f"  {i}. {step}")
            if approved.get("implementation_notes"):
                lines.append(f"- **Notes:** {approved['implementation_notes']}")
            lines.append("")
        
        # Final reasoning
        lines.append("#### 📝 Verdict Reasoning")
        lines.append("")
        lines.append(r.get("reasoning", "N/A"))
        lines.append("")
        lines.append("---")
        lines.append("")
    
    return "\n".join(lines)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Screen research papers for SciCode benchmark suitability"
    )
    parser.add_argument(
        "--pdf",
        required=True,
        help="PDF file path, directory, or space-separated list of paths"
    )
    parser.add_argument(
        "--output",
        default="screening_report.md",
        help="Output markdown report file"
    )
    parser.add_argument(
        "--model",
        default="gpt-5.2-2025-12-11",
        help="OpenAI model to use"
    )
    parser.add_argument(
        "--json-output",
        default="screening_report.json",
        help="Output JSON file with raw results"
    )
    
    args = parser.parse_args()
    
    # Initialize client
    client = OpenAI()
    
    # Collect PDF files
    pdf_files = []
    for path_str in args.pdf.split():
        path = Path(path_str.strip())
        if not path_str.strip():
            continue
        if path.is_file() and path.suffix.lower() == ".pdf":
            pdf_files.append(path)
        elif path.is_dir():
            pdf_files.extend(path.glob("*.pdf"))
        else:
            print(f"Warning: {path} not found or not a PDF")
    
    if not pdf_files:
        print("No PDF files found to screen")
        sys.exit(0)
    
    print(f"Screening {len(pdf_files)} paper(s)...")
    print()
    
    # Screen each paper
    results = []
    for pdf_path in pdf_files:
        print(f"📄 {pdf_path.name}")
        try:
            result = screen_paper(pdf_path, client, args.model)
            verdict = result.get("overall_verdict", "UNKNOWN")
            emoji = {"APPROVE": "✅", "REJECT": "❌", "NEEDS_REVIEW": "⚠️"}.get(verdict, "❓")
            print(f"  Result: {emoji} {verdict}")
        except Exception as e:
            result = {
                "_meta": {"pdf_file": pdf_path.name, "pdf_path": str(pdf_path)},
                "error": str(e)
            }
            print(f"  Result: ❌ ERROR - {e}")
        results.append(result)
        print()
    
    # Generate reports
    markdown_report = generate_markdown_report(results)
    
    Path(args.output).write_text(markdown_report)
    Path(args.json_output).write_text(json.dumps(results, indent=2))
    
    print(f"Reports written to {args.output} and {args.json_output}")
    print()
    print("=" * 60)
    print(markdown_report)


if __name__ == "__main__":
    main()


