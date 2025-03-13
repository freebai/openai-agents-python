from __future__ import annotations

import asyncio 
from dataclasses import dataclass 
from typing import Literal 

from openai import AsyncOpenAI  # 导入OpenAI异步客户端，用于与API通信
from agents import Agent, ItemHelpers, Runner, TResponseInputItem, ModelSettings, OpenAIChatCompletionsModel, set_default_openai_client

"""
这个例子展示了 LLM as a Judge 的模式。
第一个代理生成故事大纲。
第二个代理评判大纲并提供反馈。
我们循环这个过程直到评判者对大纲满意为止。
"""

# 配置参数
CONFIG = {
    "model_name_1": "qwen2.5:7b",  # 模型1
    "model_name_2": "qwq:latest",  # 模型2
    "temperature": 0.5,  # 默认温度参数，控制输出的随机性
    "api_base": "http://localhost:11434/v1",  # Ollama API地址，指向本地运行的Ollama服务
    "timeout": 120.0,  # API超时时间，单位为秒
    "max_iterations": 5,  # 最大迭代次数
}

# 设置OpenAI兼容的Ollama客户端
# 创建一个AsyncOpenAI客户端实例，但连接到本地Ollama服务器
qwen_client_1 = AsyncOpenAI(
    api_key="qwen2.5",  # Ollama不需要真实的API密钥，但API要求提供一个值
    base_url=CONFIG["api_base"],  # 使用Ollama的API地址
    timeout=CONFIG["timeout"],  # 设置超时时间
)

qwen_client_2 = AsyncOpenAI(
    api_key="qwq",  # Ollama不需要真实的API密钥，但API要求提供一个值
    base_url=CONFIG["api_base"],  # 使用Ollama的API地址
    timeout=CONFIG["timeout"],  # 设置超时时间
)


class OllamaOpenAIChatCompletionsModel(OpenAIChatCompletionsModel):
    """自定义模型类以处理Ollama API的响应格式与OpenAI API的差异"""
    
    async def stream_raw_text(self, *args, **kwargs):
        """重写流处理方法"""
        async for event in await super().stream_raw_text(*args, **kwargs):
            yield event  # 直接传递事件，如果需要可以在这里对Ollama特有的响应格式进行处理


# 代理-1 : 创建故事大纲生成器代理
story_outline_generator = Agent(
    name="story_outline_generator", 
    instructions=( 
        "你需要根据用户的输入生成一个非常简短的故事大纲。"
        "如果有任何反馈提供，请用它来改进大纲。"
    ),
    # 使用自定义Ollama模型而非默认OpenAI模型
    model=OllamaOpenAIChatCompletionsModel(  
        model=CONFIG["model_name_1"],  
        openai_client=qwen_client_1, 
    ),
    model_settings=ModelSettings(temperature=CONFIG["temperature"]), 
)


@dataclass
class EvaluationFeedback:
    """定义评估反馈的数据结构，包含反馈内容和评分"""
    feedback: str  # 反馈内容
    score: Literal["pass", "needs_improvement", "fail"]  # 评分


# 代理-2 :创建评估者代理
evaluator = Agent(
    name="evaluator",
    instructions=(
        "你需要评估一个故事大纲并决定它是否足够好。"
        "评分标准：\n"
        "- 'fail'：大纲存在重大问题，需要完全重写\n"
        "- 'needs_improvement'：大纲基本可用，但需要特定改进\n"
        "- 'pass'：大纲已经足够好\n"
        "第一次评估时给予'fail'或'needs_improvement'。"
        "第二次或更多次评估时，如果有明显改进，考虑给予更高评分。"
    ),
    # 指定输出类型为EvaluationFeedback数据类
    output_type=EvaluationFeedback,  
    model=OllamaOpenAIChatCompletionsModel( 
        model=CONFIG["model_name_2"], 
        openai_client=qwen_client_2, 
    ),
    model_settings=ModelSettings(temperature=CONFIG["temperature"]), 
)


async def main() -> None:
    """主函数"""
    msg = input("你想听什么样的故事？ ")
    # 创建输入项列表，包含用户消息
    input_items: list[TResponseInputItem] = [{"content": msg, "role": "user"}]

    # 初始化最新的故事大纲变量
    latest_outline: str | None = None

    # 初始化迭代计数器
    iteration_count = 0

    while True:
        # 增加迭代计数
        iteration_count += 1

        # 第1步：运行故事大纲生成器代理
        story_outline_result = await Runner.run(
            story_outline_generator,  # 使用故事大纲生成器代理
            input_items,  # 传入当前的输入项列表
        )

        # 第2步：更新输入项列表，包含大纲生成器的响应
        input_items = story_outline_result.to_input_list()
        # 获取最新生成的故事大纲文本
        # 一个辅助函数，用于从这些消息项中提取纯文本内容
        latest_outline = ItemHelpers.text_message_outputs(story_outline_result.new_items)
        print("故事大纲已生成")  # 打印状态信息

        # 第3步：运行评估者代理评估故事大纲
        evaluator_result = await Runner.run(
            evaluator, 
            input_items + [{"content": f"这是第{iteration_count}次迭代评估", "role": "system"}]
        )
        
        # 第4步：获取评估结果
        result: EvaluationFeedback = evaluator_result.final_output

        # 打印评估分数
        print(f"评估者评分: {result.score}")

        # 第5步：如果评分为"pass"或已达到最大迭代次数，则跳出循环
        if result.score == "pass":
            print("故事大纲已足够好，退出。")
            break
        elif iteration_count >= CONFIG["max_iterations"]:
            print(f"已达到最大迭代次数 {CONFIG['max_iterations']}，退出循环。")
            break

        # 第6步：将评估反馈添加到输入项列表，供下一轮故事大纲生成使用
        input_items.append({"content": f"反馈: {result.feedback}", "role": "user"})

    # 打印最终的故事大纲
    print(f"最终故事大纲: {latest_outline}")


# 程序入口
if __name__ == "__main__":
    asyncio.run(main())
