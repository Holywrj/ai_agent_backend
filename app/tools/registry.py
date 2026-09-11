from langchain_core.tools import BaseTool

from app.tools.weather import get_weather


def get_all_tools() -> list[BaseTool]:
    return [
        get_weather,
    ]
