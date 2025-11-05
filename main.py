"""Main entry point for RA (Robo Advisor) Agent System"""

import asyncio
import logging
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from src.agents.supervisor_agent import SupervisorAgent
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

    # Initialize supervisor agent
    supervisor = SupervisorAgent()

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
                print("\nAvailable Agents:")
                agents = supervisor.get_available_agents()
                for agent in agents:
                    print(f"\n  • {agent['name']} ({agent['type']})")
                    print(f"    {agent['description']}")
                    if "capabilities" in agent:
                        print(f"    Capabilities: {', '.join(agent['capabilities'])}")
                print()
                continue

            # Process request
            print("\nAssistant: ", end="", flush=True)
            result = await supervisor.process_request(user_input, user_id="cli_user")

            # Display response
            print(result["response"])
            print(f"\n[Task Type: {result['task_type']} | Handled by: {result['delegated_to']}]")
            print()

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
