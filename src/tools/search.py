# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

import json
from src.utils.logger import logger
import os

import requests
from langchain_community.tools import BraveSearch, DuckDuckGoSearchResults
from langchain_community.tools.arxiv import ArxivQueryRun
from langchain_community.utilities import ArxivAPIWrapper, BraveSearchWrapper
from langchain_core.tools import BaseTool
from pydantic import Field

from src.config import SearchEngine, SELECTED_SEARCH_ENGINE
from src.tools.tavily_search.tavily_search_results_with_images import (
    TavilySearchResultsWithImages,
)

from src.tools.decorators import create_logged_tool

# Create logged versions of the search tools
LoggedTavilySearch = create_logged_tool(TavilySearchResultsWithImages)
LoggedDuckDuckGoSearch = create_logged_tool(DuckDuckGoSearchResults)
LoggedBraveSearch = create_logged_tool(BraveSearch)
LoggedArxivSearch = create_logged_tool(ArxivQueryRun)


class LangSearchTool(BaseTool):
    """LangSearch web search tool (api.langsearch.com)."""

    name: str = "web_search"
    description: str = "Search the web using LangSearch API."
    max_results: int = Field(default=5)

    def _run(self, query: str) -> str:
        api_key = os.getenv("LANG_SEARCH_API_KEY", "")
        resp = requests.post(
            "https://api.langsearch.com/v1/web-search",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"query": query, "freshness": "noLimit", "summary": False, "count": self.max_results},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        results = []
        for item in data.get("data", {}).get("webPages", {}).get("value", []):
            results.append({"title": item.get("name", ""), "url": item.get("url", ""), "content": item.get("snippet", "")})
        return json.dumps(results, ensure_ascii=False)

    async def _arun(self, query: str) -> str:
        return self._run(query)


LoggedLangSearch = create_logged_tool(LangSearchTool)


# Get the selected search tool
def get_web_search_tool(max_search_results: int):
    if SELECTED_SEARCH_ENGINE == SearchEngine.TAVILY.value:
        return LoggedTavilySearch(
            name="web_search",
            max_results=max_search_results,
            include_raw_content=False,
            include_images=True,
            include_image_descriptions=True,
        )
    elif SELECTED_SEARCH_ENGINE == SearchEngine.DUCKDUCKGO.value:
        return LoggedDuckDuckGoSearch(name="web_search", max_results=max_search_results)
    elif SELECTED_SEARCH_ENGINE == SearchEngine.BRAVE_SEARCH.value:
        return LoggedBraveSearch(
            name="web_search",
            search_wrapper=BraveSearchWrapper(
                api_key=os.getenv("BRAVE_SEARCH_API_KEY", ""),
                search_kwargs={"count": max_search_results},
            ),
        )
    elif SELECTED_SEARCH_ENGINE == SearchEngine.ARXIV.value:
        return LoggedArxivSearch(
            name="web_search",
            api_wrapper=ArxivAPIWrapper(
                top_k_results=max_search_results,
                load_max_docs=max_search_results,
                load_all_available_meta=True,
            ),
        )
    elif SELECTED_SEARCH_ENGINE == SearchEngine.BOCHA.value:
        return LoggedLangSearch(name="web_search", max_results=max_search_results)
    else:
        raise ValueError(f"Unsupported search engine: {SELECTED_SEARCH_ENGINE}")


if __name__ == "__main__":
    results = LoggedDuckDuckGoSearch(
        name="web_search", max_results=3, output_format="list"
    )
    print(results.name)
    print(results.description)
    print(results.args)
    # .invoke("cute panda")
    # print(json.dumps(results, indent=2, ensure_ascii=False))
