import asyncio
from openai import AsyncOpenAI
import re
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel

from agents import Agent, Runner, trace, ModelSettings, OpenAIChatCompletionsModel, set_default_openai_client

"""
此示例演示了一个确定性流程，其中每个步骤由一个Agent执行。
1. 第一个Agent生成故事大纲
2. 我们将大纲提供给第二个Agent
3. 第二个Agent检查大纲是否质量良好，以及是否是科幻故事
4. 如果大纲质量不佳或不是科幻故事，我们就此停止
5. 如果大纲质量良好且是科幻故事，我们将大纲提供给第三个Agent
6. 第三个Agent撰写故事
7. 将最终的故事保存到本地文件
"""

# 定义要使用的Ollama模型名称
MODEL_NAME = "qwq:latest" 

# 配置参数
CONFIG = {
    "model_name": MODEL_NAME,
    "temperature": 0.5,  # 默认温度参数
    "output_dir": "deterministic_output",  # 输出目录
    "api_base": "http://localhost:11434/v1",  # Ollama API地址
    "timeout": 120.0,  # API超时时间
}

# 设置OpenAI兼容的Ollama客户端
# 创建一个AsyncOpenAI客户端实例，但连接到本地Ollama服务器
external_client = AsyncOpenAI(
    api_key="qwq", 
    base_url=CONFIG["api_base"], 
    timeout=CONFIG["timeout"],  
)

# 设置默认客户端
set_default_openai_client(external_client, use_for_tracing=False)

# 代理1: 创建故事大纲代理
# 该代理负责根据用户输入生成初步的故事大纲
story_outline_agent = Agent(
    name="story_outline_agent",  # 代理名称
    instructions="根据用户输入生成一个非常简短的故事大纲。",  # 指令
    model=OpenAIChatCompletionsModel(  # 使用OpenAI兼容的聊天完成模型
        model=CONFIG["model_name"],  # 模型名称
        openai_client=external_client,  # 使用之前配置的Ollama客户端
    ),
    model_settings=ModelSettings(temperature=CONFIG["temperature"]), 
)


# 定义大纲检查代理的输出结构
# 使用Pydantic模型确保输出格式的一致性和类型安全
class OutlineCheckerOutput(BaseModel):
    good_quality: bool  # 大纲质量是否良好
    is_scifi: bool  # 是否为科幻故事


# 代理2: 创建大纲检查代理
# 该代理负责评估故事大纲的质量并确定其类型
outline_checker_agent = Agent(
    name="outline_checker_agent",  # 代理名称
    instructions="阅读给定的故事大纲，并判断其质量。同时，确定它是否是一个科幻故事。", 
    output_type=OutlineCheckerOutput,  # 指定结构化输出类型
    model=OpenAIChatCompletionsModel(  # 使用OpenAI兼容的聊天完成模型
        model=CONFIG["model_name"],  # 模型名称
        openai_client=external_client,  # 使用之前配置的Ollama客户端
    ),
    model_settings=ModelSettings(temperature=CONFIG["temperature"]), 
)

# 代理3: 创建故事撰写代理
# 该代理负责根据给定的大纲撰写一个短篇故事
story_agent = Agent(
    name="story_agent",  # 代理名称
    instructions="根据给定的大纲撰写一个短篇故事。",
    output_type=str,
    model=OpenAIChatCompletionsModel(
        model=CONFIG["model_name"], 
        openai_client=external_client,
    ),
    model_settings=ModelSettings(temperature=CONFIG["temperature"]),
)


def save_story_to_file(story_content, user_prompt):
    """
    将生成的故事保存到本地文件
    
    Args:
        story_content: 生成的故事内容
        user_prompt: 用户的初始提示
    
    Returns:
        保存的文件路径
    """
    # 清理模型输出中的思考过程
    cleaned_story = remove_thinking_process(story_content)
    
    # 创建保存目录
    save_dir = Path(CONFIG["output_dir"])
    save_dir.mkdir(exist_ok=True)
    
    # 生成文件名 (使用时间戳确保唯一性)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}.txt"
    
    # 完整的文件路径
    file_path = save_dir / filename
    
    # 写入故事内容，包括用户提示和时间戳
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(f"用户提示: {user_prompt}\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("\n==================\n\n")
        f.write(cleaned_story)
    
    return file_path


def remove_thinking_process(text):
    """
    移除文本中<think>标签之间的内容
    
    Args:
        text: 原始文本内容
    
    Returns:
        清理后的文本
    """
    # 检查文本中是否存在<think>标签
    if "<think>" in text and "</think>" in text:
        print("检测到思考过程，正在清理...")
        # 使用正则表达式找到并移除所有<think>...</think>内容
        cleaned_text = re.sub(r'(?s)<think>.*?</think>', '', text)
        # 移除可能产生的多余空行
        cleaned_text = re.sub(r'\n{3,}', '\n\n', cleaned_text)
        return cleaned_text.strip()
    else:
        # 如果没有思考过程标签，返回原始文本
        return text


async def main():
    try:
        input_prompt = input("你想要什么类型的故事？")
        if not input_prompt.strip():
            print("提示不能为空，请重新运行程序并输入有效的提示。")
            return

        # 确保整个工作流是单个跟踪
        with trace("确定性故事流程"):
            print("正在生成故事大纲...")
            # 1. 生成大纲
            outline_result = await Runner.run(
                story_outline_agent,
                input_prompt,
            )
            print(f"已生成大纲:\n{outline_result.final_output}\n")

            # 2. 检查大纲
            print("正在检查大纲质量...")
            outline_checker_result = await Runner.run(
                outline_checker_agent,
                outline_result.final_output,
            )

            # 3. 添加一个门控，如果大纲质量不佳或不是科幻故事则停止
            assert isinstance(outline_checker_result.final_output, OutlineCheckerOutput)
            result = outline_checker_result.final_output
            
            if not result.good_quality:
                print("大纲质量不佳，到此为止。")
                return
            
            if not result.is_scifi:
                print("大纲不是科幻故事，到此为止。")
                return

            print("大纲质量良好且是科幻故事，因此我们继续撰写故事。")

            # 4. 撰写故事
            print("正在撰写故事...")
            story_result = await Runner.run(
                story_agent,
                outline_result.final_output,
            )
            
            # 故事初始版本
            current_story = story_result.final_output
            print(f"\n故事：\n{current_story}\n")
            
            # 5. 保存最终故事到本地文件
            saved_path = save_story_to_file(current_story, input_prompt)
            print(f"最终故事已保存到：{saved_path}")
            
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"发生错误: {str(e)}")


if __name__ == "__main__":
    asyncio.run(main())
