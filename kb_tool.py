from langchain_core.tools import tool

from company_kb import query_knowledgebase
from pydantic import Field

def questohive_db(query: str = Field("Your request in natural Language")):
    """Use this tool to ask about anything questohive related.
    Primarily used for customer support related questions"""
    if not query:
        return "Ask what you want in natural language"
    
    result = []
    responses = query_knowledgebase(query)
    for response in responses:
        answers = response.get("content")
        result.append(answers)
    return "\n\n".join(result)


import time
if __name__ == "__main__":
    print("Set up is complete")
    


    
