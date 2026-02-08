import json

from .base import JavaClient

class ContextTool(JavaClient):
    async def get_fm_ctx(self, function_path: str):
        return await self.get(
            "/api/context/function-context",
            {"functionPath": function_path}
        )

    async def search_fm(self, query: str):
        return await self.get(
            "/api/context/search",
            {"q": query}
        )


if __name__ == '__main__':
    import asyncio


    async def main():
        client = ContextTool()

        fm = await client.search_fm("_setComment")

        if not fm or not fm.get("success"):
            print("Search failed")
            return

        results = fm.get("results", [])
        if not results:
            print("No functions found")
            return

        first = results[0]
        path = first["path"]

        ctx = await client.get_fm_ctx(path)

        print(json.dumps(ctx, indent=2))


    asyncio.run(main())