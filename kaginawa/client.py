from typing import List
from dataclasses import dataclass
from datetime import timedelta

import requests

from kaginawa.exceptions import KaginawaError

@dataclass
class KaginawaReference:
    title: str
    snippet: str
    url: str


@dataclass
class KaginawaResponse:
    id: str
    node: str
    ms: timedelta
    api_balance: float
    output: str
    tokens: int
    references: List[KaginawaReference]


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

            print(res.json())
            res.raise_for_status()

            raw_response = res.json()
        except requests.RequestException as e:
            raise KaginawaError("Error calling /v0/fastgpt") from e

        references = [
            KaginawaReference(**ref)
            for ref in raw_response["data"]["references"]
        ]

        raw_response["data"].pop("references")

        return KaginawaResponse(
            references=references,
            **(raw_response["meta"] | raw_response["data"])
        )
