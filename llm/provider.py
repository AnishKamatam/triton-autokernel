import os
import json
import openai

class KernelGenerator:
    def __init__(self, api_key):
        self.client = openai.OpenAI(api_key=api_key)

    def generate_configs(self, n=5):
        prompt = f"""
You are a GPU optimization expert. Suggest {n} unique configurations for a Triton MatMul kernel on an A100.

Return ONLY a JSON list of dictionaries with these keys:
- BLOCK_SIZE_M (power of 2, 32 to 256)
- BLOCK_SIZE_N (power of 2, 32 to 256)
- BLOCK_SIZE_K (power of 2, 32 to 128)
- GROUP_SIZE_M (usually 8)
- num_warps (4, 8, or 16)
- num_stages (2 to 5)

Example: [{{"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8, "num_warps": 4, "num_stages": 3}}]
"""
        
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
