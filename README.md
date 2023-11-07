# Kaginawa

![project icon](project_icon.png)

An *unofficial* client to Kagi APIs

## Installation

```bash
pip install kaginawa
```

## Usage

```python
from kaginawa.client import Kaginawa

# You can also set the KAGI_API_KEY environment variable.
client = Kaginawa(token="YOUR_API_TOKEN")

response: KaginawaFastGPTResponse = client.generate(
    "Write a logstash pipeline file to send a heartbeat to a server "
    "https://example.com/heartbeat every 30 seconds"
)

print(response.output)

for reference in response.references:
    print(reference.title)
    print(reference.snippet)
    print(reference.url)

response: KaginawaEnrichResponse = client.enrich_web(query="Best fermented hot sauce")
# or 
response: KaginawaEnrichResponse = client.enrich_news(query="Is Oliver Tree okay?")

for result in response.results:
    print(result.rank)
    print(result.title)
    print(result.url)
    print(result.snippet)
    print(result.published)


response: KaginawaSummarizationResponse = client.summarize(
    url="https://example.com",
    engine=KaginawaSummarizationEngine.AGNES,
    summary_type=KaginawaSummaryType.TAKEAWAY,
    target_language="FR"
)

print(response.output)

response: KaginawaSummarizationResponse = client.summarize(
    text="The rain in Spain…",
    engine=KaginawaSummarizationEngine.CECIL
)

print(response.output)
```

## FAQ

<dl>
 <dt>Do you support the search API?</dt>
 <dd>I would love to but I don't have enterprise.</dd>

 <dt>Why the name?</dt>
 <dd>Because it's like the only word that starts with Kagi</dd>
</dl>

![kagi_meme](kagi_meme.png)

## Authors

* Estelle Poulin <dev@inspiredby.es>
