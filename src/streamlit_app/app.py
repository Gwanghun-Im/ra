"""Streamlit UI for RA (Robo Advisor) Agent System"""

import streamlit as st
import asyncio
import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(env_path)

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# from agents.supervisor_agent import SupervisorAgent
from src.a2a_agents.client import A2AClientManager
from vector_db.faiss_manager import FAISSManager
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="RA - Robo Advisor", page_icon="💼", layout="wide", initial_sidebar_state="expanded"
)


def run_async(coro):
    """Run async function in Streamlit"""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)


# Initialize session state
if "supervisor" not in st.session_state:
    st.session_state.a2a_client = run_async(
        A2AClientManager.create(remote_agent_names=["supervisor"])
    )
    st.session_state.faiss_manager = FAISSManager()
    st.session_state.messages = []
    st.session_state.user_id = "streamlit_user"


# Sidebar
with st.sidebar:
    st.title("💼 RA System")
    st.caption("Robo Advisor Agent System")

    st.divider()

    # User settings
    st.subheader("Settings")
    user_id = st.text_input("User ID", value=st.session_state.user_id)
    st.session_state.user_id = user_id

    st.divider()

    # Agent information
    st.subheader("Available Agents")

    # Get available agents asynchronously
    try:
        agents = st.session_state.a2a_client.get_discovered_agents()
        logger.info(f"Discovered agents: {agents}")
        for agent_name, agent_info in agents.items():
            with st.expander(f"**{agent_name}**"):
                st.write(f"*Description:* {agent_info.get('description', 'N/A')}")
    except Exception as e:
        st.error(f"Error loading agents: {e}")

    st.divider()

    # System status
    st.subheader("System Status")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Messages", len(st.session_state.messages))
    with col2:
        faiss_stats = st.session_state.faiss_manager.get_stats()
        st.metric("Vectors", faiss_stats["total_vectors"])

    st.divider()

    # Clear conversation
    if st.button("Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# Main content
st.title("🤖 Robo Advisor Chat")
st.caption("Ask questions about your portfolio, investments, and market insights")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        # Display metadata if available
        if "metadata" in message:
            with st.expander("Details"):
                st.json(message["metadata"])

# Chat input
if prompt := st.chat_input("Ask me anything about investments..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get agent response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Process request through supervisor
                task_result = run_async(
                    st.session_state.a2a_client.send_task(
                        agent_name="supervisor",
                        message=prompt,
                        task_id=f"cli_task_{hash(prompt)}",
                    )
                )

                if task_result is None:
                    st.error("Failed to get response from agent")
                    st.session_state.messages.append(
                        {"role": "assistant", "content": "Failed to get response from agent"}
                    )
                else:
                    # Extract response text from Task object
                    # Task object has artifacts which contain the actual response
                    response_text = "No response available"
                    if hasattr(task_result, "artifacts") and task_result.artifacts:
                        for artifact in task_result.artifacts:
                            if hasattr(artifact, "parts"):
                                for part in artifact.parts:
                                    if hasattr(part, "text"):
                                        response_text = part.text
                                        break

                    # Display response
                    st.markdown(response_text)

                    # Display task info
                    task_info = {
                        "task_id": task_result.id if hasattr(task_result, "id") else "unknown",
                        "status": (
                            task_result.status if hasattr(task_result, "status") else "unknown"
                        ),
                    }

                    with st.expander("Task Information"):
                        st.json(task_info)

                    # Add assistant message
                    st.session_state.messages.append(
                        {"role": "assistant", "content": response_text, "metadata": task_info}
                    )

            except Exception as e:
                error_msg = f"Error: {str(e)}"
                st.error(error_msg)
                logger.error(f"Error processing request: {e}")

                st.session_state.messages.append({"role": "assistant", "content": error_msg})

st.divider()
st.caption("RA - Robo Advisor Agent System | Powered by MCP & A2A")
