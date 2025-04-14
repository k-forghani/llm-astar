"""
Prompts designed for DeepSeek models.
Contains different prompt techniques - standard, chain-of-thought (CoT), 
and reflexion prompting (ReAct) variants.
"""

# Parsing prompt for DeepSeek
PARSE_DEEPSEEK = """<|im_start|>system
You are a code generation assistant. Your task is to convert natural language descriptions of pathfinding problems into structured JSON objects that can be used as input for a pathfinding algorithm. The input will describe start and goal points, horizontal and vertical barriers, and the boundaries of the environment. 

For each input, extract the following information and format it into a JSON object:
1. "start": The starting point coordinates as [x, y].
2. "goal": The goal point coordinates as [x, y].
3. "range_x": The minimum and maximum x-coordinates of the environment range as [x_min, x_max].
4. "range_y": The minimum and maximum y-coordinates of the environment range as [y_min, y_max].
5. "horizontal_barriers": A list of horizontal barriers, each represented as [y, x_start, x_end].
6. "vertical_barriers": A list of vertical barriers, each represented as [x, y_start, y_end].

Ensure the JSON object is properly formatted, matches the required structure, and values are correctly extracted. (Do not return any comment, just the JSON)
<|im_end|>

<|im_start|>user
find the shortest path from (5, 5) to (45, 25) on a 51x31 grid, avoiding horizontal barriers at [15, 10-21] and vertical barriers at [20, 0-15], [30, 15-30], and [40, 0-16].
<|im_end|>

<|im_start|>assistant
{"start": [5, 5], "goal": [45, 25], "range_x": [0, 51], "range_y": [0, 31], "horizontal_barriers": [[15, 10, 21]], "vertical_barriers": [[20, 0, 15], [30, 15, 30], [40, 0, 16]]}
<|im_end|>

<|im_start|>user
design a path from [29, 4] to [22, 21] on a 51 by 31 grid that avoids horizontal barriers centered at [[7, 21, 33]] and vertical barriers at [[16, 7, 31], [39, 3, 23], [1, 24, 24]].
<|im_end|>

<|im_start|>assistant
{'start': [29, 4], 'goal': [22, 21], 'range_x': [0, 51], 'range_y': [0, 31], 'horizontal_barriers': [[7, 21, 33]], 'vertical_barriers': [[16, 7, 31], [39, 3, 23], [1, 24, 24]]}
<|im_end|>

<|im_start|>user
{query}
<|im_end|>

<|im_start|>assistant
"""

# Standard prompting technique
STANDARD_DEEPSEEK = """<|im_start|>system
You need to find the shortest path between the start and goal points that navigates around barriers. 
Horizontal barriers are represented as [y, x_start, x_end], and vertical barriers are represented as [x, y_start, y_end].
Conclude your response with the generated path in the format "Generated Path: [[x1, y1], [x2, y2], ...]".
<|im_end|>

<|im_start|>user
Start Point: [5, 5]
Goal Point: [20, 20]
Horizontal Barriers: [[10, 0, 25], [15, 30, 50]]
Vertical Barriers: [[25, 10, 22]]
<|im_end|>

<|im_start|>assistant
Generated Path: [[5, 5], [26, 9], [25, 23], [20, 20]]
<|im_end|>

<|im_start|>user
Start Point: [2, 3]
Goal Point: [18, 8]
Horizontal Barriers: [[7, 0, 10], [5, 30, 40]]
Vertical Barriers: [[14, 12, 30], [40, 0, 11]] 
<|im_end|>

<|im_start|>assistant
Generated Path: [[2, 3], [11, 6], [18, 8]]
<|im_end|>

<|im_start|>user
Start Point: [40, 20]
Goal Point: [10, 10]
Horizontal Barriers: [[15, 20, 30], [15, 35, 50]]
Vertical Barriers: [[20, 4, 30]]
<|im_end|>

<|im_start|>assistant
Generated Path: [[40, 20], [31, 15], [20, 3], [10, 10]]
<|im_end|>

<|im_start|>user
Start Point: {start}
Goal Point: {goal}
Horizontal Barriers: {horizontal_barriers}
Vertical Barriers: {vertical_barriers}
<|im_end|>

<|im_start|>assistant
"""

# Chain-of-thought prompting technique
COT_DEEPSEEK = """<|im_start|>system
You need to find the shortest path between the start and goal points that navigates around barriers. 
Horizontal barriers are represented as [y, x_start, x_end], and vertical barriers are represented as [x, y_start, y_end].
Conclude your response with the generated path in the format "Generated Path: [[x1, y1], [x2, y2], ...]".
<|im_end|>

<|im_start|>user
Start Point: [5, 5]
Goal Point: [20, 20]
Horizontal Barriers: [[10, 0, 25], [15, 30, 50]]
Vertical Barriers: [[25, 10, 22]]
<|im_end|>

<|im_start|>assistant
Thought: Identify a path from [5, 5] to [20, 20] while avoiding the horizontal barrier at y=10 spanning x=0 to x=25 by moving upwards and right, then bypass the vertical barrier at x=25 spanning y=10 to y=22, and finally move directly to [20, 20].
Generated Path: [[5, 5], [26, 9], [25, 23], [20, 20]]
<|im_end|>

<|im_start|>user
Start Point: [2, 3]
Goal Point: [18, 8]
Horizontal Barriers: [[7, 0, 10], [5, 30, 40]]
Vertical Barriers: [[14, 12, 30], [40, 0, 11]] 
<|im_end|>

<|im_start|>assistant
Thought: Navigate from [2, 3] to [18, 8] by moving upward-right to avoid the horizontal barrier at y=7 spanning x=0 to x=10, then proceed directly to [18, 8].
Generated Path: [[2, 3], [11, 6], [18, 8]]
<|im_end|>

<|im_start|>user
Start Point: [25, 25]
Goal Point: [4, 4]
Horizontal Barriers: [[7, 0, 20], [15, 10, 15]]
Vertical Barriers: [[10, 15, 30], [30, 0, 18], [10, 0, 5]]
<|im_end|>

<|im_start|>assistant
Thought: Move from [25, 25] to [4, 4] by heading downward to bypass the horizontal barrier at y=7 spanning x=0 to x=20, and finally move directly to [4, 4], where bypass the vertical barrier at x=10 spanning y=0 to y=5.
Generated Path: [[25, 25], [21, 7], [10, 6], [4, 4]]
<|im_end|>

<|im_start|>user
Start Point: {start}
Goal Point: {goal}
Horizontal Barriers: {horizontal_barriers}
Vertical Barriers: {vertical_barriers}
<|im_end|>

<|im_start|>assistant
"""

# Reflexion prompting technique
REPE_DEEPSEEK = """<|im_start|>system
You need to find the shortest path between the start and goal points that navigates around barriers. 
Horizontal barriers are represented as [y, x_start, x_end], and vertical barriers are represented as [x, y_start, y_end].
Conclude your response with the generated path in the format "Generated Path: [[x1, y1], [x2, y2], ...]".
<|im_end|>

<|im_start|>user
Start Point: [5, 5]
Goal Point: [20, 20]
Horizontal Barriers: [[10, 0, 25], [15, 30, 50]]
Vertical Barriers: [[25, 10, 22]]
<|im_end|>

<|im_start|>assistant
- First Iteration on [5, 5]
Thought: The horizontal barrier at y=10 spanning x=0 to x=25 blocks the direct path to the goal. To navigate around it, we should move to the upper-right corner of the barrier.
Selected Point: [26, 9]
Evaluation: The selected point [26, 9] effectively bypasses the horizontal barrier, positioning us at its corner and maintaining progress toward the goal without encountering additional obstacles.
- Second Iteration on [26, 9]
Thought: Now that we have bypassed the horizontal barrier, the path to the goal seems clear.
Selected Point: [20, 20]
Evaluation: The path is obstructed by the vertical barrier, leading to a collision. A more effective route involves moving around this vertical barrier.
Thought: To bypass the vertical barrier at x=25, we should move along its length and then turn around it to continue toward the goal.
Selected Point: [25, 23]
Evaluation: The selected point [25, 23] successfully avoids the vertical barrier and brings us closer to the goal without encountering further obstacles.
- Third Iteration on [25, 23]
Thought: From this position, there are no barriers directly obstructing the path to the goal.
Selected Point: [20, 20]
Evaluation: The path to the goal is clear from here, allowing a direct move to the goal.
Generated Path: [[5, 5], [26, 9], [25, 23], [20, 20]]
<|im_end|>

<|im_start|>user
Start Point: [2, 3]
Goal Point: [18, 8]
Horizontal Barriers: [[7, 0, 10], [5, 30, 40]]
Vertical Barriers: [[14, 12, 30], [40, 0, 11]] 
<|im_end|>

<|im_start|>assistant
- First Iteration on [2, 3]
Thought: The horizontal barrier at y=7 spanning x=0 to x=10 obstructs the direct path. To bypass it, move to the end of the barrier.
Selected Point: [11, 6]
Evaluation: The point [11, 6] navigates around the horizontal barrier, positioning us at its edge and closer to the goal.
- Second Iteration on [11, 6]
Thought: With the barrier behind us, the path to the goal is clear, allowing us to proceed directly to the goal.
Selected Point: [18, 8]
Evaluation: The path is unobstructed from here to the goal, enabling a straightforward move to [18, 8].
Generated Path: [[2, 3], [11, 6], [18, 8]]
<|im_end|>

<|im_start|>user
Start Point: {start}
Goal Point: {goal}
Horizontal Barriers: {horizontal_barriers}
Vertical Barriers: {vertical_barriers}
<|im_end|>

<|im_start|>assistant
"""

# Add additional utility prompts similar to the ones in Qwen prompts
AGENT_DEEPSEEK = """<|im_start|>system
You are a helpful AI assistant with expertise in solving problems step by step.
<|im_end|>

<|im_start|>user
{query}

Your task is to solve this problem. You should think through the problem step by step and find the solution.

CONTEXT:
{context}
<|im_end|>

<|im_start|>assistant
"""

SUMMARY_DEEPSEEK = """<|im_start|>system
You are a helpful AI assistant that can create summaries from text.
<|im_end|>

<|im_start|>user
Summarize the following information:

{text}
<|im_end|>

<|im_start|>assistant
"""

PLANNING_DEEPSEEK = """<|im_start|>system
You are a helpful AI assistant specializing in planning complex tasks.
<|im_end|>

<|im_start|>user
I need to solve the following problem:

{query}

Create a step-by-step plan to approach this problem efficiently.
<|im_end|>

<|im_start|>assistant
"""

# Mapping of prompt types to actual prompt templates
DEEPSEEK_PROMPTS = {
    'standard': STANDARD_DEEPSEEK,
    'cot': COT_DEEPSEEK,
    'repe': REPE_DEEPSEEK,
    'agent': AGENT_DEEPSEEK,
    'summary': SUMMARY_DEEPSEEK,
    'planning': PLANNING_DEEPSEEK
} 