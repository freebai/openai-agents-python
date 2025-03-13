import asyncio
import requests
import json
import os
from typing import Optional, Dict, Any, Tuple
from agents import Agent, Runner, ModelSettings, OpenAIChatCompletionsModel, set_default_openai_client
from openai import AsyncOpenAI
from pydantic import BaseModel
from agents import Agent, Runner, function_tool

# 定义要使用的Ollama模型名称
MODEL_NAME = "qwq:latest"  # 使用本地部署的Ollama模型

# 配置参数
CONFIG = {
    "model_name": MODEL_NAME,  # 模型名称
    "temperature": 0.5,  # 默认温度参数，控制输出的随机性
    "api_base": "http://localhost:11434/v1",  # Ollama API地址，指向本地运行的Ollama服务
    "timeout": 120.0,  # API超时时间，单位为秒
}

# 高德地图API配置
AMAP_KEY = "3a0fcbd868ab659bb41f2b110676edcf"  # 替换为你的高德API Key

# 高德地图天气查询API接口地址，用于获取城市天气信息
AMAP_WEATHER_URL = "https://restapi.amap.com/v3/weather/weatherInfo"

# 高德地图地理编码API接口地址，用于将城市名称转换为地理编码（adcode）
AMAP_GEOCODE_URL = "https://restapi.amap.com/v3/geocode/geo"

# 城市编码缓存文件
CITY_CODE_CACHE_FILE = os.path.join(os.path.dirname(__file__), "city_code_cache.json")

# 设置OpenAI兼容的Ollama客户端
# 创建一个AsyncOpenAI客户端实例，但连接到本地Ollama服务器
external_client = AsyncOpenAI(
    api_key="qwq",  # Ollama不需要真实的API密钥，但API要求提供一个值
    base_url=CONFIG["api_base"],  # 使用Ollama的API地址
    timeout=CONFIG["timeout"],  # 设置超时时间
)

# 设置默认客户端，用于所有代理的通信
set_default_openai_client(external_client, use_for_tracing=False)


def load_city_code_cache() -> Dict[str, str]:
    """加载城市编码缓存"""
    if os.path.exists(CITY_CODE_CACHE_FILE):
        try:
            with open(CITY_CODE_CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return {}


def save_city_code_cache(cache: Dict[str, str]) -> None:
    """保存城市编码缓存"""
    try:
        with open(CITY_CODE_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
    except IOError:
        pass


def get_city_code(city_name: str) -> Tuple[Optional[str], str]:
    """
    获取城市的adcode编码
    
    Args:
        city_name: 城市名称
        
    Returns:
        Tuple[Optional[str], str]: (城市编码, 错误信息)
    """
    # 先检查缓存
    city_code_cache = load_city_code_cache()
    if city_name in city_code_cache:
        return city_code_cache[city_name], ""
    
    # 调用高德地理编码API获取城市编码
    try:
        params = {
            "key": AMAP_KEY,
            "address": city_name,
        }
        response = requests.get(AMAP_GEOCODE_URL, params=params)
        response.raise_for_status()
        
        data = response.json()
        if data["status"] == "1" and data["count"] != "0":
            # 获取第一个结果的adcode
            adcode = data["geocodes"][0]["adcode"]
            
            # 缓存结果
            city_code_cache[city_name] = adcode
            save_city_code_cache(city_code_cache)
            
            return adcode, ""
        else:
            return None, f"无法找到城市 '{city_name}' 的编码"
            
    except Exception as e:
        return None, f"获取城市编码时发生错误: {str(e)}"


@function_tool
def get_weather(city: str) -> str:
    """
    使用高德地图API获取指定城市的天气信息
    
    Args:
        city: 城市名称，如"北京"、"上海"、"深圳"等
        
    Returns:
        str: 格式化的天气信息字符串
    """
    try:
        # 获取城市编码
        city_code, error_msg = get_city_code(city)
        if error_msg:
            return f"获取城市编码失败: {error_msg}"
        if not city_code:
            return f"无法获取'{city}'的天气信息，高德API仅支持中国城市，请尝试输入中国的城市名称。"
        
        # 使用城市编码查询天气
        params = {
            "key": AMAP_KEY,
            "city": city_code,
            "extensions": "all"  # 获取预报和实况天气
        }
        
        response = requests.get(AMAP_WEATHER_URL, params=params)
        response.raise_for_status()
        
        data = response.json()
        
        # 检查API返回状态
        if data["status"] == "1":
            # 获取实况天气
            if "lives" in data and len(data["lives"]) > 0:
                live_weather = data["lives"][0]
                
                # 第1步：提取天气信息
                weather = live_weather["weather"]  # 天气现象
                temperature = live_weather["temperature"]  # 温度
                humidity = live_weather["humidity"]  # 湿度
                wind_direction = live_weather["winddirection"]  # 风向
                wind_power = live_weather["windpower"]  # 风力
                report_time = live_weather["reporttime"]  # 数据发布时间
                
                # 格式化返回信息
                weather_info = (
                    f"城市: {city}\n"
                    f"天气: {weather}\n"
                    f"温度: {temperature}°C\n"
                    f"湿度: {humidity}%\n"
                    f"风向: {wind_direction}\n"
                    f"风力: {wind_power}\n"
                    f"数据更新时间: {report_time}\n"
                )
                
                # 第2步：获取天气预报
                if "forecasts" in data and len(data["forecasts"]) > 0:
                    forecast = data["forecasts"][0]
                    if "casts" in forecast and len(forecast["casts"]) > 0:
                        weather_info += "\n未来天气预报:\n"
                        for i, cast in enumerate(forecast["casts"][:3]):  # 最多显示3天预报
                            date = cast["date"]
                            day_weather = cast["dayweather"]
                            night_weather = cast["nightweather"]
                            day_temp = cast["daytemp"]
                            night_temp = cast["nighttemp"]
                            day_wind = f"{cast['daywind']}风 {cast['daypower']}级"
                            night_wind = f"{cast['nightwind']}风 {cast['nightpower']}级"
                            
                            weather_info += (
                                f"日期: {date}\n"
                                f"白天: {day_weather}, {day_temp}°C, {day_wind}\n"
                                f"夜间: {night_weather}, {night_temp}°C, {night_wind}\n"
                            )
                
                return weather_info
            else:
                return f"未找到{city}的实时天气信息。"
        else:
            return f"无法获取{city}的天气信息，API返回错误。"
            
    except requests.exceptions.RequestException as e:
        return f"天气API请求失败: {str(e)}"
    except (KeyError, ValueError) as e:
        return f"解析天气数据失败: {str(e)}"
    except Exception as e:
        return f"获取天气信息时发生未知错误: {str(e)}"


@function_tool
def get_weather_brief(city: str) -> str:
    """
    使用高德地图API获取指定城市的简要天气信息
    
    Args:
        city: 城市名称，如"北京"、"上海"等
        
    Returns:
        str: 简要的天气信息字符串
    """
    try:
        # 获取城市编码
        city_code, error_msg = get_city_code(city)
        if error_msg:
            return f"获取城市编码失败: {error_msg}"
        if not city_code:
            return f"无法获取'{city}'的天气信息，高德API仅支持中国城市。"
        
        # 使用城市编码查询天气
        params = {
            "key": AMAP_KEY,
            "city": city_code,
            "extensions": "base"  # 只获取实况天气
        }
        
        response = requests.get(AMAP_WEATHER_URL, params=params)
        response.raise_for_status()
        
        data = response.json()
        
        # 检查API返回状态
        if data["status"] == "1" and data["count"] != "0":
            weather_info = data["lives"][0]
            
            # 提取天气信息
            weather = weather_info["weather"]  # 天气现象
            temperature = weather_info["temperature"]  # 温度
            
            # 格式化返回简要信息
            return f"{city}当前天气: {weather}, 温度{temperature}°C"
        else:
            return f"无法获取{city}的天气信息。"
            
    except Exception as e:
        return f"获取天气信息时发生错误: {str(e)}"


def create_weather_agent(model_name: str = MODEL_NAME, client: Optional[AsyncOpenAI] = None) -> Agent:
    """
    创建天气助手代理
    
    Args:
        model_name: 要使用的模型名称，默认为全局配置的MODEL_NAME
        client: 可选的AsyncOpenAI客户端，默认使用全局配置的external_client
        
    Returns:
        Agent: 配置好的天气助手代理实例
    """
    return Agent(
        name="天气助手",
        instructions="你是一个提供天气信息的助手，使用高德地图API获取实时天气数据。你可以提供详细的天气信息或简要的天气概况。高德API只支持中国城市的天气查询。",
        tools=[get_weather, get_weather_brief],
        model=OpenAIChatCompletionsModel(
            model=model_name, 
            openai_client=client or external_client,
        ),
        model_settings=ModelSettings(temperature=0.5),
    )


# 创建默认的天气助手实例
agent = create_weather_agent()


async def main():
    """主函数，用于测试天气查询功能"""
    # 1. 测试获取中国城市的天气
    result = await Runner.run(agent, input="深圳的天气预报详情怎么样？")
    print(result.final_output)
    print("-"*100)
    
    # 2. 测试简要天气查询
    result = await Runner.run(agent, input="给我深圳的天气简报")
    print(result.final_output)
    print("-"*100)
    
    # 3. 测试非中国城市
    result = await Runner.run(agent, input="纽约的天气怎么样？")
    print(result.final_output)
    print("-"*100)

if __name__ == "__main__":
    asyncio.run(main())