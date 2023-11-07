from datetime import datetime, timedelta

from kaginawa.models import (
    KaginawaEnrichWebResponse,
    KaginawaFastGPTResponse,
    KaginawaResponse,
    KaginawaSearchResult,
    KaginawaSummarizationResponse,
)


class TestKaginawaModels:
    """Test Kagi dataclasses.

    The only thing that's actually necessary to test is whether they're
    able to parse valid Kagi responses. These tests serve as a reference of
    what format the project *expects* the responses to be so it's easy to
    determine if the code needs updated due to upstream changes."
    """

    def test_kagi_response(self):
        common_kagi_response = {
            "meta": {
                "id": (id := "cbad1ea2-ddef-42e7-af75-81bce2c64924"),
                "node": (node := "us-central1"),
                "ms": (ms := 1),
                "api_balance": (api_balance := 5.0),
            }
        }

        res = KaginawaResponse.from_raw(common_kagi_response)

        assert res.id == id
        assert res.node == node
        assert res.duration == timedelta(milliseconds=ms)
        assert res.api_balance == api_balance

    def test_kagi_fastgpt_response(self):
        fastgpt_kagi_response = {
            "meta": {
                "id": (id := "cbad1ea2-ddef-42e7-af75-81bce2c64924"),
                "node": (node := "us-central1"),
                "ms": (ms := 1),
                "api_balance": (api_balance := 5.0),
            },
            "data": {
                "output": (output := ""),
                "tokens": (tokens := 15),
                "references": (
                    references := [
                        {
                            "title": "Top 10 Cutest Puppies",
                            "snippet": "The definitive list.",
                            "url": "https://example.com/cute-puppies.html",
                        },
                        {
                            "title": "100 Puppers You Don't Want To Miss",
                            "snippet": "You'll wish you were in a pupper puddle.",
                            "url": "https://example.com/recent/puppers",
                        },
                    ]
                ),
            },
        }

        res = KaginawaFastGPTResponse.from_raw(fastgpt_kagi_response)

        assert res.id == id
        assert res.node == node
        assert res.duration == timedelta(milliseconds=ms)
        assert res.api_balance == api_balance

        assert res.output == output
        assert res.tokens == tokens

        assert len(res.references) == len(references)

        for parsed_reference, raw_reference in list(zip(res.references, references)):
            assert parsed_reference.title == raw_reference["title"]
            assert parsed_reference.snippet == raw_reference["snippet"]
            assert parsed_reference.url == raw_reference["url"]

    def test_kagi_fastgpt_response_no_references(self):
        fastgpt_kagi_response = {
            "meta": {
                "id": (id := "cbad1ea2-ddef-42e7-af75-81bce2c64924"),
                "node": (node := "us-central1"),
                "ms": (ms := 1),
                "api_balance": (api_balance := 5.0),
            },
            "data": {
                "output": (output := ""),
                "tokens": (tokens := 15),
            },
        }

        res = KaginawaFastGPTResponse.from_raw(fastgpt_kagi_response)

        assert res.id == id
        assert res.node == node
        assert res.duration == timedelta(milliseconds=ms)
        assert res.api_balance == api_balance

        assert res.output == output
        assert res.tokens == tokens
        assert len(res.references) == 0

    def test_kagi_enrich_web_response(self):
        enrich_web_response = {
            "meta": {
                "id": (id := "cbad1ea2-ddef-42e7-af75-81bce2c64924"),
                "node": (node := "us-central1"),
                "ms": (ms := 1),
                "api_balance": (api_balance := 5.0),
            },
            "data": (
                results := [
                    {
                        "t": 42,
                        "rank": 0,
                        "url": "https://example.com/which-animal-are-you.html",
                        "title": "Quiz: Which Animal Are You?",
                        "snippet": "Maybe a cat, maybe a dog, find out now!",
                        "published": (datetime.now() - timedelta(days=5)).isoformat(),
                    },
                ]
            ),
        }

        res = KaginawaEnrichWebResponse.from_raw(enrich_web_response)

        assert res.id == id
        assert res.node == node
        assert res.duration == timedelta(milliseconds=ms)
        assert res.api_balance == api_balance

        assert len(res.results) == len(results)

        for parsed_result, raw_result in list(zip(res.results, results)):
            assert parsed_result.title == raw_result["title"]
            assert parsed_result.snippet == raw_result["snippet"]
            assert parsed_result.url == raw_result["url"]

    def test_kagi_summarization_response(self):
        summarization_kagi_response = {
            "meta": {
                "id": (id := "cbad1ea2-ddef-42e7-af75-81bce2c64924"),
                "node": (node := "us-central1"),
                "ms": (ms := 1),
                "api_balance": (api_balance := 5.0),
            },
            "data": {
                "tokens": (tokens := 42),
                "output": (output := "tl;dr War and Peace: It involves Russia."),
            },
        }

        res = KaginawaSummarizationResponse.from_raw(summarization_kagi_response)

        assert res.id == id
        assert res.node == node
        assert res.duration == timedelta(milliseconds=ms)
        assert res.api_balance == api_balance

        assert res.tokens == tokens
        assert res.output == output

    def test_kagi_search_result(self):
        kagi_search_result = {
            "t": (t := 37),
            "rank": (rank := 0),
            "url": (url := "https://example.com/best-ya-romance-novels-this-year"),
            "title": (title := "Straignt from TikTok: This Year's Best YA Novels"),
            "snippet": (snippet := "Necromancer lesbians and gay love in ancient Greece!"),
            "published": (published := (datetime.now() - timedelta(weeks=7)).isoformat()),
        }

        res = KaginawaSearchResult.from_raw(kagi_search_result)

        assert res.t == t
        assert res.rank == rank
        assert res.title == title
        assert res.url == url
        assert res.snippet == snippet
        assert res.published == datetime.fromisoformat(published)
