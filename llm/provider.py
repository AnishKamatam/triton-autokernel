import os
import json
import openai

class KernelGenerator:
    def __init__(self, api_key):
        self.client = openai.OpenAI(api_key=api_key)

    def generate_configs(self, n=5, history_context=None, kernel_type="matmul"):
        if kernel_type == "fused":
            base_prompt = f"""
You are a GPU optimization expert. Suggest {n} unique configurations for a Triton Fused MatMul+Activation kernel on an A100.

Return ONLY a JSON list of dictionaries with these keys:
- BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K (power of 2, 32 to 256)
- GROUP_SIZE_M (usually 8)
- ACTIVATION ("relu" or "leaky_relu")
- num_warps (4, 8, or 16)
- num_stages (2 to 5)

Example: [{{"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8, "ACTIVATION": "leaky_relu", "num_warps": 4, "num_stages": 3}}]
"""
        else:
            base_prompt = f"""
You are a GPU optimization expert. Suggest {n} unique configurations for a Triton MatMul kernel on an A100.

Return ONLY a JSON list of dictionaries with these keys:
- BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K (power of 2, 32 to 256)
- GROUP_SIZE_M (usually 8)
- num_warps (4, 8, or 16)
- num_stages (2 to 5)
"""

        if history_context:
            feedback_prompt = f"""

### PREVIOUS PERFORMANCE DATA:

{history_context}

### INSTRUCTIONS:

1. Analyze the 'error' entries (like shared memory OOM). Avoid those parameter combinations.
2. Look at the 'success' entries with high TFLOPS. Try to refine those (e.g., tweak num_stages or num_warps).
3. Do not suggest configs you have already tried.
"""
            prompt = base_prompt + feedback_prompt
        else:
            prompt = base_prompt

        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}]
        )
        
        content = response.choices[0].message.content.strip()
        if content.startswith("```json"):
            content = content[7:-3].strip()
        elif content.startswith("```"):
            content = content[3:-3].strip()
            
        return json.loads(content)
