from langchain_core.tools import tool


@tool
def get_weather(city: str) -> str:
    """查询指定城市的天气"""
    return f'{city}今天晴天，25度。'
