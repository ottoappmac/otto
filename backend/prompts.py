"""LLM generation prompts for agent and skill creation, and summarization."""

AGENT_GENERATION_PROMPT = """\
You are an agent specification generator. Given the user's description of what they \
want an agent to do, produce a JSON object with the following keys:

- "name": a kebab-case identifier (e.g. "sap-order-creator")
- "description": 1-2 sentence summary of what the agent does
- "system_prompt": a complete AGENTS.md-style markdown document with sections: \
  Role, Core Workflow, Key Rules, Speed, Output Conventions, Learned Preferences
- "tools": list of MCP server IDs this agent needs (e.g. ["playwright-mcp"])
- "skills": list of skill names this agent should use (e.g. ["playwright-browser"])

Respond ONLY with valid JSON. No markdown fences, no explanation."""

SKILL_GENERATION_PROMPT = """\
You are a skill document generator. Given the user's description of domain knowledge \
or procedural expertise, produce a complete SKILL.md document with:

1. YAML frontmatter (---) containing: name, description
2. Markdown body with sections: When to use, Steps/Workflow, Critical Rules, Prerequisites

The document should follow this exact format:
---
name: skill-name
description: One-line description
---

# Skill Title

## When to use
...

## Steps
...

## Critical Rules
...

## Prerequisites
...

Respond ONLY with the SKILL.md content. No markdown fences around the whole response."""

# ---------------------------------------------------------------------------
# Structured compact / summary template
# Adapted from Claude Code's two-phase analysis → summary approach.
# The <analysis> block is a drafting scratchpad that is stripped by
# `format_compact_summary()` before the summary reaches the context window.
# ---------------------------------------------------------------------------

_ANALYSIS_INSTRUCTION = """\
Before providing your final summary, wrap your analysis in <analysis> tags to organize your thoughts. In your analysis:

1. Chronologically walk through each message / section. For each, identify:
   - The user's explicit requests and intents
   - Your approach to addressing them
   - Key decisions, technical concepts, and code patterns
   - Specific details: file names, code snippets, function signatures, file edits
   - Errors encountered and how they were resolved
   - Pay special attention to user feedback, especially corrections
2. Double-check for technical accuracy and completeness."""

STRUCTURED_SUMMARY_PROMPT = f"""\
Your task is to create a detailed summary of the conversation so far, paying \
close attention to the user's explicit requests and your previous actions. \
This summary must be thorough enough that someone reading only it can continue \
development without losing context.

{_ANALYSIS_INSTRUCTION}

Your summary should include the following sections:

1. Primary Request and Intent: Capture all of the user's explicit requests and intents in detail.
2. Key Technical Concepts: List all important technical concepts, technologies, and frameworks discussed.
3. Files and Code Sections: Enumerate specific files and code sections examined, modified, or created. Include full code snippets where applicable and a summary of why each file is important.
4. Errors and Fixes: List all errors encountered and how they were fixed. Include user feedback on errors.
5. Problem Solving: Document problems solved and any ongoing troubleshooting efforts.
6. All User Messages: List ALL user messages that are not tool results. These are critical for understanding changing intent.
7. Pending Tasks: Outline any pending tasks you have explicitly been asked to work on.
8. Current Work: Describe precisely what was being worked on immediately before this summary request, including file names and code snippets.
9. Optional Next Step: List the next step directly in line with the user's most recent explicit request. If the last task was concluded, only list next steps that are explicitly requested. Include direct quotes from the most recent conversation showing exactly what task you were working on.

Structure your response as:

<analysis>
[Your thought process ensuring all points are covered]
</analysis>

<summary>
1. Primary Request and Intent:
   [Detailed description]

2. Key Technical Concepts:
   - [Concept 1]
   - [Concept 2]

3. Files and Code Sections:
   - [File Name 1]
      - [Why this file is important]
      - [Changes made, if any]
      - [Important Code Snippet]

4. Errors and Fixes:
   - [Error description]:
     - [How you fixed it]
     - [User feedback if any]

5. Problem Solving:
   [Description of solved problems and ongoing troubleshooting]

6. All User Messages:
   - [Detailed non-tool-use user message]

7. Pending Tasks:
   - [Task 1]
   - [Task 2]

8. Current Work:
   [Precise description of current work]

9. Optional Next Step:
   [Next step to take]
</summary>

Please provide your summary now, following this structure."""


# ---------------------------------------------------------------------------
# Lite compact prompt for OSS-local summarizers.
#
# The full ``STRUCTURED_SUMMARY_PROMPT`` above asks the model to do a
# two-phase analysis → summary in 9 sections, which works well on Claude
# but blows out latency and quality on smaller open-source models.  Lite
# drops the ``<analysis>`` scratchpad and trims to the 5 sections that
# actually drive resumption: the request, the files touched, pending
# tasks, current work, and the next step.
#
# ``format_compact_summary()`` works on either output because the lite
# prompt still uses the same ``<summary>...</summary>`` envelope.
# ---------------------------------------------------------------------------

STRUCTURED_SUMMARY_PROMPT_LITE = """\
Summarize the conversation so far so that someone reading only the summary \
can continue without losing context.

Wrap the answer in <summary>...</summary> tags using these sections:

1. Primary Request and Intent: the user's explicit requests in 2-4 sentences.
2. Files and Code: list every file examined / created / modified with a one-line
   note on why it matters. Include critical code snippets where useful.
3. Pending Tasks: anything the user explicitly asked for that isn't done yet.
4. Current Work: precisely what was being worked on right before this summary.
5. Next Step: the single next action that follows from the user's most recent
   message; quote the user verbatim where helpful.

Format:

<summary>
1. Primary Request and Intent:
   [...]

2. Files and Code:
   - path/to/file.py — [why it matters]

3. Pending Tasks:
   - [...]

4. Current Work:
   [...]

5. Next Step:
   [...]
</summary>

Provide the summary now."""


def format_compact_summary(summary: str) -> str:
    """Strip the <analysis> scratchpad and unwrap <summary> tags.

    The analysis block improves summary quality but has no informational value
    once the summary is written.
    """
    import re

    formatted = summary

    # Strip analysis section
    formatted = re.sub(r"<analysis>[\s\S]*?</analysis>", "", formatted)

    # Extract and unwrap summary section
    match = re.search(r"<summary>([\s\S]*?)</summary>", formatted)
    if match:
        content = match.group(1).strip()
        formatted = re.sub(r"<summary>[\s\S]*?</summary>", f"Summary:\n{content}", formatted)

    # Collapse excessive blank lines
    formatted = re.sub(r"\n\n+", "\n\n", formatted)

    return formatted.strip()
