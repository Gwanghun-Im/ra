"""Main entry point for RA (Robo Advisor) Agent System"""

import asyncio
import logging
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from src.a2a.client import A2AClient
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("logs/ra_system.log")],
)

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


async def main():
    """Main function to run the RA system"""
    logger.info("Starting RA (Robo Advisor) Agent System")

    # Initialize A2A client to communicate with Supervisor
    a2a_client = A2AClient()

    print("\n" + "=" * 60)
    print("   RA - Robo Advisor Agent System")
    print("=" * 60)
    print("\nAvailable commands:")
    print("  - Type your question or request")
    print("  - Type 'agents' to see available agents")
    print("  - Type 'exit' or 'quit' to exit")
    print("\n" + "=" * 60 + "\n")

    # Interactive loop
    while True:
        try:
            # Get user input
            user_input = input("You: ").strip()

            if not user_input:
                continue

            # Check for exit commands
            if user_input.lower() in ["exit", "quit", "q"]:
                print("\nGoodbye! 👋")
                break

            # Special command to show agents
            if user_input.lower() == "agents":
                print("\nDiscovering available agents...")
                try:
                    # Discover agents via A2A protocol
                    discovered = await a2a_client.get_card()
                    print(f"\nFound {len(discovered)} agent(s):\n")
                    for agent_card in discovered:
                        print(f"  • {agent_card.get('name', 'Unknown')}")
                        print(f"    {agent_card.get('description', '')}")
                        if "capabilities" in agent_card:
                            print(f"    Capabilities: {', '.join(agent_card['capabilities'])}")
                        print(f"    URL: {agent_card.get('service_url', '')}")
                        print()
                except Exception as e:
                    print(f"\n❌ Error discovering agents: {e}\n")
                continue

            # Process request via A2A protocol
            print("\nAssistant: ", end="", flush=True)
            try:
                # Send task to supervisor agent via A2A
                result = await a2a_client.send_task(
                    agent_name="supervisor",
                    message=user_input,
                    task_id=f"cli_task_{hash(user_input)}",
                    context={"user_id": "cli_user", "source": "cli"},
                )

                # Extract and display response
                if "message" in result:
                    response = result["message"].get("content", "")
                else:
                    response = result.get("response", str(result))

                print(response)

                # Display metadata if available
                if "artifacts" in result:
                    artifacts = result["artifacts"]
                    for artifact in artifacts:
                        if artifact.get("type") == "routing_info":
                            data = artifact.get("data", {})
                            print(
                                f"\n[Task Type: {data.get('task_type', 'unknown')} | Handled by: {data.get('delegated_to', 'unknown')}]"
                            )
                print()

            except Exception as e:
                print(f"Error: {str(e)}\n")

        except KeyboardInterrupt:
            print("\n\nInterrupted by user. Exiting...")
            break

        except Exception as e:
            logger.error(f"Error processing request: {e}")
            print(f"\n❌ Error: {str(e)}\n")

    logger.info("RA System shutdown complete")


if __name__ == "__main__":
    # Ensure logs directory exists
    Path("logs").mkdir(exist_ok=True)

    # Run main function
    try:
        asyncio.run(main())
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)
