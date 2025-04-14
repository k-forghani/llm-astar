PARSE_DEEPSEEK = """<|im_start|>system
You are a helpful AI assistant. Your task is to extract information from a graph search result.
<|im_end|>

<|im_start|>user
{query}

The nodes included in the search are:
{nodes}

Extract the useful information from these nodes that can help answering the query.
<|im_end|>

<|im_start|>assistant
"""

DEEPSEEK_PROMPTS = {
    "agent": """<|im_start|>system
You are a helpful AI assistant with expertise in solving problems step by step.
<|im_end|>

<|im_start|>user
{query}

Your task is to solve this problem. You should think through the problem step by step and find the solution.

CONTEXT:
{context}
<|im_end|>

<|im_start|>assistant
""",

    "summary": """<|im_start|>system
You are a helpful AI assistant that can create summaries from text.
<|im_end|>

<|im_start|>user
Summarize the following information:

{text}
<|im_end|>

<|im_start|>assistant
""",

    "planning": """<|im_start|>system
You are a helpful AI assistant specializing in planning complex tasks.
<|im_end|>

<|im_start|>user
I need to solve the following problem:

{query}

Create a step-by-step plan to approach this problem efficiently.
<|im_end|>

<|im_start|>assistant
"""
} 