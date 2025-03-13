# Configuring the SDK

## API keys and clients
# 配置API密钥和客户端
# 默认情况下SDK会从环境变量OPENAI_API_KEY获取API密钥

# 方法1：直接设置默认API密钥
from agents import set_default_openai_key
set_default_openai_key("sk-...")  # 替换为你的实际API密钥

# 方法2：自定义OpenAI客户端
from openai import AsyncOpenAI
from agents import set_default_openai_client
custom_client = AsyncOpenAI(
    base_url="...",  # 自定义API端点
    api_key="..."    # 自定义API密钥
)
set_default_openai_client(custom_client)  # 设置自定义客户端

# 方法3：选择使用的OpenAI API类型
from agents import set_default_openai_api
set_default_openai_api("chat_completions")  # 使用聊天补全API

## Tracing
# 追踪功能配置
# 默认启用，使用与LLM相同的API密钥

# 设置专门的追踪API密钥
from agents import set_tracing_export_api_key
set_tracing_export_api_key("sk-...")  # 设置追踪专用密钥

# 禁用追踪功能
from agents import set_tracing_disabled
set_tracing_disabled(True)  # 传入True禁用追踪

## Debug logging
# 调试日志配置
# 默认只输出警告和错误

# 启用详细日志输出
from agents import enable_verbose_stdout_logging
enable_verbose_stdout_logging()  # 启用详细日志

# 自定义日志配置
import logging
logger = logging.getLogger("openai.agents")  # 获取SDK日志器
logger.setLevel(logging.DEBUG)  # 设置日志级别
logger.addHandler(logging.StreamHandler())  # 添加控制台处理器

### Sensitive data in logs
# 敏感数据日志控制

# 禁用模型数据日志
export OPENAI_AGENTS_DONT_LOG_MODEL_DATA=1

# 禁用工具数据日志
export OPENAI_AGENTS_DONT_LOG_TOOL_DATA=1
