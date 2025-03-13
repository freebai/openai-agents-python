import asyncio  # 导入异步IO库，用于支持异步操作

from openai import AsyncOpenAI  # 导入OpenAI异步客户端
from openai.types.responses import ResponseContentPartDoneEvent, ResponseTextDeltaEvent  # 导入响应类型定义

# 导入agents库中的核心组件
from agents import Agent, ItemHelpers, MessageOutputItem, Runner, trace, ModelSettings, OpenAIChatCompletionsModel, set_default_openai_client

"""
该示例展示了"代理作为工具"模式。前线代理接收用户消息，然后选择调用哪些代理作为工具。
在此例中，它从一组翻译代理中进行选择。
"""

# 定义要使用的Ollama模型名称
MODEL_NAME = "qwq:latest"  # 使用本地部署的Ollama模型

# 配置参数
CONFIG = {
    "model_name": MODEL_NAME,  # 模型名称
    "temperature": 0.5,  # 默认温度参数，控制输出的随机性
    "api_base": "http://localhost:11434/v1",  # Ollama API地址，指向本地运行的Ollama服务
    "timeout": 120.0,  # API超时时间，单位为秒
}

# 设置OpenAI兼容的Ollama客户端
# 创建一个AsyncOpenAI客户端实例，但连接到本地Ollama服务器
external_client = AsyncOpenAI(
    api_key="qwq",  # Ollama不需要真实的API密钥，但API要求提供一个值
    base_url=CONFIG["api_base"],  # 使用Ollama的API地址
    timeout=CONFIG["timeout"],  # 设置超时时间
)

# 设置默认客户端，用于所有代理的通信
set_default_openai_client(external_client, use_for_tracing=False)

# 自定义 OpenAIChatCompletionsModel 类来处理 Ollama 的响应格式
class OllamaOpenAIChatCompletionsModel(OpenAIChatCompletionsModel):
    """自定义模型类以处理Ollama API的响应格式与OpenAI API的差异"""
    
    async def stream_raw_text(self, *args, **kwargs):
        """重写流处理方法，处理可能的属性差异"""
        async for event in await super().stream_raw_text(*args, **kwargs):
            yield event  # 直接传递事件，如果需要可以在这里对Ollama特有的响应格式进行处理

# 创建西班牙语翻译代理
spanish_agent = Agent(
    name="spanish_agent",  # 代理名称
    instructions="你负责将用户的消息翻译成西班牙语",  # 代理的指令
    handoff_description="一个英语到西班牙语的翻译器",  # 代理的描述，用于在工具调用时
    model=OllamaOpenAIChatCompletionsModel(  # 使用自定义Ollama模型
        model=CONFIG["model_name"], 
        openai_client=external_client,
    ),
    model_settings=ModelSettings(temperature=CONFIG["temperature"]),  # 模型设置
)

# 创建法语翻译代理
french_agent = Agent(
    name="french_agent",
    instructions="你负责将用户的消息翻译成法语",
    handoff_description="一个英语到法语的翻译器",
    model=OllamaOpenAIChatCompletionsModel(
        model=CONFIG["model_name"], 
        openai_client=external_client,
    ),
    model_settings=ModelSettings(temperature=CONFIG["temperature"]),
)

# 创建意大利语翻译代理
italian_agent = Agent(
    name="italian_agent",
    instructions="你负责将用户的消息翻译成意大利语",
    handoff_description="一个英语到意大利语的翻译器",
    model=OllamaOpenAIChatCompletionsModel(
        model=CONFIG["model_name"], 
        openai_client=external_client,
    ),
    model_settings=ModelSettings(temperature=CONFIG["temperature"]),
)

# 创建中文翻译代理
chinese_agent = Agent(
    name="chinese_agent",
    instructions="你负责将用户的消息翻译成中文",
    handoff_description="一个英语到中文的翻译器",
    model=OllamaOpenAIChatCompletionsModel(
        model=CONFIG["model_name"], 
        openai_client=external_client,
    ),
    model_settings=ModelSettings(temperature=CONFIG["temperature"]),
)

# 创建英语翻译代理
english_agent = Agent(
    name="english_agent",
    instructions="你负责将用户的消息翻译成英语",
    handoff_description="一个其他语言到英语的翻译器",
    model=OllamaOpenAIChatCompletionsModel(
        model=CONFIG["model_name"], 
        openai_client=external_client,
    ),
    model_settings=ModelSettings(temperature=CONFIG["temperature"]),
)

# 创建协调器代理，负责选择和调用合适的翻译代理
orchestrator_agent = Agent(
    name="orchestrator_agent",
    instructions=(
        "你是一个翻译代理。你使用提供给你的工具进行翻译。"
        "如果被要求进行多种翻译，你将按顺序调用相关工具。"
        "你永远不要自己翻译，而是始终使用提供的工具。"
    ),
    tools=[  # 将各个翻译代理注册为工具
        spanish_agent.as_tool(
            tool_name="translate_to_spanish",  # 工具名称
            tool_description="将用户的消息翻译成西班牙语",  # 工具描述
        ),
        french_agent.as_tool(
            tool_name="translate_to_french",
            tool_description="将用户的消息翻译成法语",
        ),
        italian_agent.as_tool(
            tool_name="translate_to_italian",
            tool_description="将用户的消息翻译成意大利语",
        ),
        chinese_agent.as_tool(
            tool_name="translate_to_chinese",
            tool_description="将用户的消息翻译成中文",
        ),
        english_agent.as_tool(
            tool_name="translate_to_english",
            tool_description="将用户的消息翻译成英语",
        ),
    ],
    model=OllamaOpenAIChatCompletionsModel(
        model=CONFIG["model_name"], 
        openai_client=external_client,
    ),
    model_settings=ModelSettings(temperature=CONFIG["temperature"]),
)

# 创建合成代理，用于检查翻译结果并生成最终响应
synthesizer_agent = Agent(
    name="synthesizer_agent",
    instructions="你负责检查翻译，在需要时进行更正，并生成最终的合并响应。",
    model=OllamaOpenAIChatCompletionsModel(
        model=CONFIG["model_name"], 
        openai_client=external_client,
    ),
    model_settings=ModelSettings(temperature=CONFIG["temperature"]),
)


async def main():
    """主函数，处理用户输入并运行代理系统"""
    try:
        # 获取用户输入
        msg = input("你好！你想翻译什么内容，以及翻译成哪些语言？")

        # 在单个跟踪中运行整个编排过程
        with trace("编排评估器"):  # 开始一个跟踪块，用于性能监控或日志记录
            # 运行协调器代理处理用户输入
            orchestrator_result = await Runner.run(orchestrator_agent, msg)

            # 输出协调器的处理步骤
            for item in orchestrator_result.new_items:
                if isinstance(item, MessageOutputItem):
                    text = ItemHelpers.text_message_output(item)
                    if text:
                        print(f"  --- 翻译步骤: --- \n {text}")

            # 将协调器的结果传递给合成代理进行最终处理
            synthesizer_result = await Runner.run(
                synthesizer_agent, orchestrator_result.to_input_list()
            )
        print("--------------------------------")
        
    except KeyboardInterrupt:
        # 处理用户中断（如Ctrl+C）
        print("\n程序被用户中断")
    except Exception as e:
        # 处理其他异常
        print(f"发生错误: {str(e)}")


# 程序入口点，当脚本直接运行时执行
if __name__ == "__main__":
    asyncio.run(main())  # 使用asyncio运行异步主函数
