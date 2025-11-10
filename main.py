"""Main entry point for RA (Robo Advisor) Agent System"""

import asyncio
import logging
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from src.a2a_agents.client import A2AClientManager
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
    a2a_client = await A2AClientManager.create(remote_agent_names=["supervisor"])

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
                    discovered = a2a_client.get_discovered_agents()
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
                task_result = await a2a_client.send_message(
                    agent_name="supervisor",
                    message=user_input,
                    task_id=f"cli_task_{hash(user_input)}",
                )

                if task_result is None:
                    print("Failed to get response from agent\n")
                    continue

                # Extract response text from Task object
                response = "No response available"
                if hasattr(task_result, "artifacts") and task_result.artifacts:
                    for artifact in task_result.artifacts:
                        if hasattr(artifact, "parts"):
                            for part in artifact.parts:
                                if hasattr(part, "text"):
                                    response = part.text
                                    break

                print(response)

                # Display task metadata
                if hasattr(task_result, "id"):
                    print(f"\n[Task ID: {task_result.id}", end="")
                    if hasattr(task_result, "status"):
                        print(f" | Status: {task_result.status}", end="")
                    print("]")
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
