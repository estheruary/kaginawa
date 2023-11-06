import os
from typing import Optional

import requests

from kaginawa.exceptions import KaginawaError
from kaginawa.models import (
    KaginawaEnrichWebResponse,
    KaginawaFastGPTResponse,
    KaginawaSummarizationResponse,
)


class Kaginawa:
    """The main client to the Kagi API"""

    def __init__(
        self,
        token: Optional[str] = None,
        session: Optional[requests.Session] = None,
        api_base: str = "https://kagi.com/api",
    ):
        """Create a new instance of the Kagi API wrapper.

        Parameters:
          token (Optional[str], optional): The API access token to authenticate
            requests. If this value is unset the library will attempt to read the
            environment variable KAGI_API_KEY. If both are unset an exception will
            be raised.

          session (Optional[requests.Session], optional): An optional `requests`
            session object to use for sending HTTP requests. Defaults to `None`.

          api_base (str, optional): The base URL for the Kagi API endpoint.
            Defaults to "https://kagi.com/api".
        """

        try:
            token = token or os.environ["KAGI_API_KEY"]
        except KeyError as e:
            raise KaginawaError("Value for `token` not given and env KAGI_API_KEY is unset") from e

        self.token = token
        self.api_base = api_base

        if not session:
            session = requests.Session()

        self.session = session

        self.session.headers = {"Authorization": f"Bot {self.token}"}

    def generate(self, query: str, cache: Optional[bool] = None):
        """Generate a FastGPT response from a text query.

        Parameters:
          query (str): The prompt to send to FastGPT.

          cache (Optional[bool], optional): Allow cached responses. Defaults to `None`.

        Returns:
          KaginawaFastGPTResponse: A model representing the response.
        """
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
        """Query the Teclis index for relevant web results for a given query.

        Parameters:
          query (str): The search query.

        Returns:
          KaginawaEnrichWebResponse: A model representing the response.
        """

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
        """Summarize a URL or text snippet.

        Parameters:
          url (Optional[str], optional): The URL to summarize. This option is exclusive
            with the `text` parameter. Defaults to `None`.

          text (Optional[str], optional): The text snippet to summarize. This option is
            exclusive with the `url` parameter. Defaults to `None`.

          engine (Optional[str], optional): The summarization engine to use. There is a
            helper enum `KaginawaSummarizationEngine` with the valid values for this
            parameter. Defaults to `None`.

          summary_type (Optional[str], optional): The 'kind' of summary you would like
            the model to use. There is a helper enum KaginawaSummaryType with the valid
            values for this parameter. Defaults to `None`.

          target_language: (Optional[str], optional): The language CODE (eg. EN, ZH, FR)
            corresponding to the language you would like the summary in. See
            https://help.kagi.com/kagi/api/summarizer.html for valid language codes.
            Defaults to `None`.

          cache (Optional[bool], optional): Allow cached responses. Defaults to `None`.

        Returns:
          KaginawaSummarizationResponse: A model representing the response.
        """
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
