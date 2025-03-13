import asyncio
from openai import AsyncOpenAI
from agents import Agent, Runner, ModelSettings, OpenAIChatCompletionsModel, set_default_openai_client

async def main():
    # 设置自定义OpenAI客户端
    custom_client = AsyncOpenAI(
     # 自定义API端点
    base_url="https://ark.cn-beijing.volces.com/api/v3", 
    # 自定义API密钥
    api_key="c9e7d8b1-ca29-4c1d-85bb-68fa9d399a6d"    
    )
    set_default_openai_client(custom_client,use_for_tracing=False)

    agent = Agent(
        name="Assistant",
        instructions="You only respond in haikus.",
        model=OpenAIChatCompletionsModel(
            model="ep-20250217151433-6xcvv",
            openai_client=custom_client,
        ),
        # model_settings=ModelSettings(temperature=0.5)
    )

    result = await Runner.run(agent, "你好")
    print(result.final_output)
    # Tell me about recursion in programming.
    # Function calls itself,
    # Looping in smaller pieces,
    # Endless by design.


if __name__ == "__main__":
    asyncio.run(main())
