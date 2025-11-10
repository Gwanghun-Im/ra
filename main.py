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


def extract_task_response(stream_response):
    """Extract response text from a streaming response (Task tuple or Message).

    Args:
        stream_response: Either a (Task, UpdateEvent) tuple or Message object

    Returns:
        Tuple of (response_text, task_object) or (None, None) if no text found
    """
    # Handle tuple format: (Task, UpdateEvent)
    if isinstance(stream_response, tuple):
        task, _ = stream_response

        if hasattr(task, "artifacts") and task.artifacts:
            for artifact in task.artifacts:
                if hasattr(artifact, "parts"):
                    for part in artifact.parts:
                        actual_part = part.root if hasattr(part, "root") else part
                        if hasattr(actual_part, "text"):
                            return actual_part.text, task

    return None, None


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

            # Process request via A2A protocol with streaming
            print("\nAssistant: ", end="", flush=True)
            try:
                # Send task to supervisor agent via A2A with streaming enabled
                final_response = None
                final_task = None

                async for stream_response in a2a_client.send_message(
                    agent_name="supervisor",
                    message=user_input,
                    context_id=str(hash("cli_user")),
                    metadata={"user_id": "cli_user"},
                    streaming=True,
                ):
                    response_text, task = extract_task_response(stream_response)

                    if response_text:
                        # Clear previous line and display new response
                        print("\r" + " " * 100 + "\r", end="")
                        print(f"Assistant: {response_text}", end="", flush=True)
                        final_response = response_text
                        final_task = task

                # Final newline and metadata
                if final_response:
                    print()
                    if final_task and hasattr(final_task, "id"):
                        print(f"\n[Task ID: {final_task.id}", end="")
                        if hasattr(final_task, "status"):
                            status = (
                                final_task.status.state
                                if hasattr(final_task.status, "state")
                                else final_task.status
                            )
                            print(f" | Status: {status}", end="")
                        print("]")
                else:
                    print("No response available")
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
