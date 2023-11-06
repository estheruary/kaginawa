from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum

import requests

from kaginawa.exceptions import KaginawaError


@dataclass
class KaginawaReference:
    title: str
    """The <title> of the reference."""

    snippet: str
    """A short blurb of text describing the reference."""

    url: str
    """The URL of the reference."""


@dataclass
class KaginawaResponse:
    id: str
    """The unique id of the API response."""

    node: str
    """The 'node' which is really a region the API response came from."""

    duration: timedelta
    """How long the request took to process (always 0 as of 2023-11-05)."""

    api_balance: float
    """How much money (in dollars) you have left on your account."""

    @classmethod
    def from_raw(cls, raw_response: Dict[str, Any], **kwargs):
        duration = timedelta(milliseconds=raw_response["meta"]["ms"])
        raw_response["meta"].pop("ms")

        return cls(
            duration=duration,
            **raw_response["meta"],
            **kwargs,
        )
        

@dataclass
class KaginawaFastGPTResponse(KaginawaResponse):
    output: str
    """The output of FastGPT."""

    tokens: int
    """The number of the tokens in the response (always 0 as of 2023-11-05)."""

    references: List[KaginawaReference]
    """A list of web pages that were provided as context to the LLM."""

    @classmethod
    def from_raw(cls, raw_response, **kwargs):
        references = [
            KaginawaReference(**ref)
            for ref in raw_response["data"]["references"]
        ]

        raw_response["data"].pop("references")

        return super().from_raw(
            raw_response,
            references=references,
            **raw_response["data"],
            **kwargs,
        )


@dataclass
class KaginawaSearchResult:
    t: int
    """Undocumented field of unknown purpose."""

    rank: int
    """The rank of the search result (0 is most relevant)."""

    url: str
    """The URL of the search result."""

    title: str
    """The <title> of the page referenced."""

    snippet: str
    """A short blurb of text describing the result."""

    published: datetime
    """The (self-reported) date the result was published."""

    @classmethod
    def from_raw(cls, raw_result: Dict[str, Any]):
        _raw_result_copy = raw_result.copy()

        published = datetime.fromisoformat(_raw_result_copy["published"])
        _raw_result_copy.pop("published")

        return cls(
            published=published,
            **_raw_result_copy
        )


@dataclass
class KaginawaEnrichWebResponse(KaginawaResponse):
    results: List[KaginawaSearchResult]

    @classmethod
    def from_raw(cls, raw_response: Dict[str, Any], **kwargs):
        results = [
            KaginawaSearchResult.from_raw(raw_result)
            for raw_result in raw_response["data"]
        ]

        return super().from_raw(
            results=results,
            **kwargs
        )


@dataclass
class KaginawaSummarizationResponse(KaginawaResponse):
    tokens: int
    """The number of tokens in the response."""

    output: str
    """The summary produced by the API."""

    @classmethod
    def from_raw(cls, raw_response: Dict[str, Any], **kwargs):
        return super().from_raw(
            raw_response,
            tokens=raw_response["data"]["tokens"],
            output=raw_response["data"]["output"],
            **kwargs,
        )


class KaginawaSummarizationEngine(StrEnum):
    CECIL = "cecil"
    """Friendly, descriptive, fast summary."""

    AGNES = "agnes"
    """Formal, technical, analytical summary."""

    DAPHNE = "daphne"
    """Informal, creative, friendly summary."""

    MURIEL = "muriel"
    """Best-in-class summary using 'enterprise-grade' model."""


class KaginawaSummaryType(StrEnum):
    SUMMARY = "summary"
    """Paragraph(s) of summary prose."""

    TAKEAWAY = "takeaway"
    """Bulleted list of key points."""


class Kaginawa:
    def __init__(
        self,
        token: str,
        session: requests.Session | None = None,
        api_base: str = "https://kagi.com/api",
    ):
        self.token = token
        self.api_base = api_base

        if not session:
            session = requests.Session()

        self.session = session

        self.session.headers = {
            "Authorization": f"Bot {self.token}"
        }


    def generate(self, query: str, cache: bool = True):
        try:
            res = self.session.post(
                f"{self.api_base}/v0/fastgpt",
                json={
                    "query": query,
                    "cache": cache,
                },
            )

            res.raise_for_status()

            raw_response = res.json()
        except requests.RequestException as e:
            raise KaginawaError("Error calling /v0/fastgpt") from e

        return KaginawaFastGPTResponse.from_raw(raw_response)


    def enrich_web(self, query: str):
        try:
            res = self.session.get(
                f"{self.api_base}/v0/enrich/web",
                params={"q": query},
            )
            res.raise_for_status()

            raw_response = res.json()
        except requests.RequestException as e:
            raise KaginawaError("Error calling /v0/enrich/web") from e

        return KaginawaEnrichWebResponse.from_raw(raw_response)

    def summarize(
        self,
        url: Optional[str] = None,
        text: Optional[str] = None,
        engine: Optional[str] = None,
        summary_type: Optional[str] = None,
        target_language: Optional[str] = None,
        cache: Optional[bool] = None,
    ):
        try:
            params = {}

            if not (bool(url) ^ bool(text)):
                raise KaginawaError("You must provide exactly one of 'url' or 'text'.")

            if url:
                params["url"] = url

            if text:
                params["text"] = text

            if engine:
                params["engine"] = engine

            if summary_type:
                params["summary_type"] = summary_type

            if target_language:
                params["target_language"] = target_language

            if cache is not None:
                params["cache"] = cache

            res = self.session.post(
                f"{self.api_base}/v0/summarize",
                data=params,
            )

            raw_response = res.json()
        except requests.RequestException as e:
            raise KaginawaError("Error calling /v0/summarize") from e

        return KaginawaSummarizationResponse.from_raw(raw_response)
