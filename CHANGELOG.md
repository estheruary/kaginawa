# CHANGELOG

# 0.0.8

Async Support!

* We now use the `httpx` package to provide sync and asyn support.

```python
import asyncio

from kaginawa import AsyncKaginawa


async def amain():
    kagi_client = AsyncKaginawa(...)
    res = await kagi_client.generate(...)
    print(res.output)

if __name__ == "__main__":
    asyncio.run(amain())
```
