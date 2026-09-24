from langchain_openai import ChatOpenAI

from app.core.config import settings


def create_llm(
        thinking: bool = True
) -> ChatOpenAI:
    # Thinking Mode 不支持 tool_choice=required 或指定具体工具, (deepseek)
    return ChatOpenAI(
        model=settings.deepseek_chat_model,
        api_key=settings.deepseek_api_key,
        base_url=settings.deepseek_base_url,
        extra_body={
            'thinking': {
                'type': 'enabled' if thinking else 'disabled'
            }
        }
    )
