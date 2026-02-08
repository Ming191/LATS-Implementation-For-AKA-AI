import httpx

class JavaClient:
    def __init__(self, base_url="http://localhost:8080"):
        self.base_url = base_url

    async def _get(self, path: str, params: dict = None):
        async with httpx.AsyncClient() as client:
            r = await client.get(f"{self.base_url}{path}", params=params)
            r.raise_for_status()
            return r.json()

    async def get(self, path: str, params=None):
        return await self._get(path, params)

    async def _post(self, path: str, params: dict = None):
        async with httpx.AsyncClient() as client:
            r = await client.post(f"{self.base_url}{path}", json=params)
            r.raise_for_status()
            return r.json()

    async def post(self, path: str, params=None):
        return await self._post(path, params)
