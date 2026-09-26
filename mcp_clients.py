
from langchain_core.tools import tool
from vector_store import search_with_scores
from pydantic import Field

@tool
async def database(request:str = Field("Use this arguement to request for any pastquestions you want from the database")) -> None:
    """
    Use this tool to search for any past question you want from the questohive database.
    It returns the result of the top 4 closest to your search
    """

    results = await search_with_scores(request, k=4
    )
    for i, r in enumerate(results, start=1):
        print(f"--- Result {i} (score: {r['score']}) ---")
        return f"{r["content"]} Result:{i}"


if __name__ == "__main__":
    print("Set up is complete")
