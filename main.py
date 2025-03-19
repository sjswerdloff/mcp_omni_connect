import asyncio
from client import MCPClient, Configuration
from cli import MCPClientCLI

async def main():
    config = Configuration()
    client = MCPClient(config)
    cli = MCPClientCLI(client)
    try:
        await client.connect_to_servers()
        await cli.chat_loop()
    finally:
        await client.cleanup()  

if __name__ == "__main__":
    asyncio.run(main())