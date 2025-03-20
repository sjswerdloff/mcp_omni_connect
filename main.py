import asyncio
from client import MCPClient, Configuration
from cli import MCPClientCLI
from utils import logger, setup_platform


platform_config = setup_platform()

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
        logger.info("Shutting down client...")
        await client.cleanup()
        logger.info("Client shut down successfully")

if __name__ == "__main__":
    if platform_config["is_windows"]:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())