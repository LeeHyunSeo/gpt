from typing import Type
from langchain.chat_models import ChatOpenAI
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from langchain.agents import initialize_agent, AgentType
from langchain.utilities import WikipediaAPIWrapper
from langchain.utilities import GoogleSearchAPIWrapper

llm = ChatOpenAI(temperature=0.1, model_name="gpt-3.5-turbo-0125")


class WikipediaResearchToolArgsSchema(BaseModel):
    query: str = Field(
        description="The query you will research website content for search results. Example query: Research for Apple Company"
    )


class WikipediaResearchTool(BaseTool):
    name = "ResearchTool"
    description = """
    Use this tool to research on questions.
    It takes a query as an argument.
    """
    args_schema: Type[
        WikipediaResearchToolArgsSchema
    ] = WikipediaResearchToolArgsSchema

    def _run(self, query):
        wiki = WikipediaAPIWrapper()
        return wiki.run(query)

class GoogleResearchToolArgsSchema(BaseModel):
    query: str = Field(
        description="The query you will research website content for search results. Example query: Research for Apple Company"
    )


class GoogleResearchTool(BaseTool):
    name = "ResearchTool"
    description = """
    Use this tool to research on questions.
    It takes a query as an argument.
    """
    args_schema: Type[
        GoogleResearchToolArgsSchema
    ] = GoogleResearchToolArgsSchema

    def _run(self, query):
        google = GoogleSearchAPIWrapper()
        return google.run(query)

agent = initialize_agent(
    llm=llm,
    verbose=True,
    agent=AgentType.OPENAI_FUNCTIONS,
    handle_parsing_errors=True,
    tools=[
        WikipediaResearchTool(),
        # GoogleResearchTool()
    ],
)

prompt = "Research about the XZ backdoor"


docs = agent.invoke(prompt)
print(docs)