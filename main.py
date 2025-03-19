import asyncio
from client import MCPClient, Configuration
from cli import MCPClientCLI
from utils import logger


async def main():
    try:
        config = Configuration()
        client = MCPClient(config)
        cli = MCPClientCLI(client)
        await client.connect_to_servers()
        await cli.chat_loop()
    except KeyboardInterrupt:
        logger.info("Shutting down client...")
        await client.cleanup()
        logger.info("Client shut down successfully")
        return
    except Exception as e:
        logger.error(f"Error: {e}")
    finally:
        await client.cleanup()  

if __name__ == "__main__":
    asyncio.run(main())