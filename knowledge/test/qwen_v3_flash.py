from openai import OpenAI
import os
from dotenv import load_dotenv
load_dotenv()
print(os.getenv("OPENAI_API_KEY"))
# 初始化OpenAI客户端
client = OpenAI(
    api_key = "sk-72c8c186db364a148f7634cb31878002",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)


# 创建聊天完成请求
completion = client.chat.completions.create(
    model="qwen3-vl-flash",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://img.alicdn.com/imgextra/i1/O1CN01gDEY8M1W114Hi3XcN_!!6000000002727-0-tps-1024-406.jpg"
                    },
                },
                {"type": "text", "text": "这道题怎么解答？"},
            ],
        },
    ],


    # 解除以下注释会在最后一个chunk返回Token使用量
    # stream_options={
    #     "include_usage": True
    # }
)


response = completion.choices[0].message.content

print(response)

# print("=" * 20 + "完整思考过程" + "=" * 20 + "\n")
# print(reasoning_content)
# print("=" * 20 + "完整回复" + "=" * 20 + "\n")
# print(answer_content)