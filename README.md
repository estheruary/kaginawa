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

client = Kaginawa(token="YOUR_API_TOKEN")

response: KaginawaResponse = client.generate(
    "Write a logstash pipeline file to send a heartbeat to a server "
    "https://example.com/heartbeat every 30 seconds"
)

print(response.output)

for reference in response.references:
    print(reference.title)
    print(reference.snippet)
    print(reference.url)
```