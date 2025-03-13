import os
from openai import OpenAI

client = OpenAI(
    # 从环境变量中读取您的方舟API Key
    api_key="c9e7d8b1-ca29-4c1d-85bb-68fa9d399a6d",   # 自定义API密钥
    base_url="https://ark.cn-beijing.volces.com/api/v3",
    )
completion = client.chat.completions.create(
    # 将推理接入点 <Model>替换为 Model ID
    model="ep-20250217151433-6xcvv",
    messages=[
        {"role": "user", "content": "你好"}
    ]
)
print(completion.choices[0].message)