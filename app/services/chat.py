from app.core.llm import create_llm


async def chat(
        message: str,
        user_id: int
) -> str:
    llm = create_llm()

    response = await llm.ainvoke(message)

    return response.content
