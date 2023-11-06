from typing import Optional

import requests

from kaginawa.exceptions import KaginawaError
from kaginawa.models import (
    KaginawaEnrichWebResponse,
    KaginawaFastGPTResponse,
    KaginawaSummarizationResponse,
)


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

        self.session.headers = {"Authorization": f"Bot {self.token}"}

    def generate(self, query: str, cache: Optional[bool] = None):
        try:
            optional_params = {}

            if cache is not None:
                optional_params["cache"] = cache

            res = self.session.post(
                f"{self.api_base}/v0/fastgpt",
                json={
                    "query": query,
                    **optional_params,
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
