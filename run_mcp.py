import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient


async def validate_github_mcp():
    """
    Valida se o GitHub MCP Server está:
    - iniciando corretamente
    - conectando
    - expondo tools
    """

    try:
        client = MultiServerMCPClient(
            {
                "github": {
                    "command": "npx",
                    "args": ["@modelcontextprotocol/server-github"],
                    "transport": "stdio",
                    "env": {
                        # Substitua pelo seu token
                        "GITHUB_PERSONAL_ACCESS_TOKEN": ""
                    },
                }
            }
        )

        print("Conectando ao MCP server...\n")

        tools = await client.get_tools()

        print("✅ MCP server conectado com sucesso!\n")

        print(f"Quantidade de tools encontradas: {len(tools)}\n")

        print("Tools disponíveis:\n")

        for tool in tools:
            print(f"- {tool.name}")

        print("\n✅ Validação concluída.")

    except Exception as e:
        print("\n❌ Erro ao conectar no MCP server:\n")
        print(type(e).__name__)
        print(str(e))


if __name__ == "__main__":
    asyncio.run(validate_github_mcp())