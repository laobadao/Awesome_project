
from openai import OpenAI
client = OpenAI(base_url ="https://api.gptsapi.net",
                api_key="sk-Zij62df3b23042c795acecf69a481f54069b709735380hfx")
# client.base_url = "https://c65049dc8af4456cd77f7ed6d4299d3b.api-forwards.com"
# client.base_url = "https://api.gptsapi.net"
# API 转发
# https://c65049dc8af4456cd77f7ed6d4299d3b.api-forwards.com

# 将 OpenAI API 基础地址 https://api.openai.com 替换为 https://api.gptsapi.net ，复制下面的密钥即可使用。
# 将 Claude API 基础地址 https://api.anthropic.com 替换为 https://api.gptsapi.net ，复制下面的密钥即可使用。


stream = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Say this is a test"}],
    stream=True,
)
for chunk in stream:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="")