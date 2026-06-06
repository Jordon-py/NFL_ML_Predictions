"""
Interactive Loop Script using the Google Antigravity Python SDK.

This script implements both the Basic Interactive Loop and the Advanced Conversation
Flow provided by the google-antigravity SDK. It includes clean command-line options
to select the mode, automated environment validation, and robust platform guides
for running on unsupported OSes like native Windows.

Usage:
    python scripts/interactive_loop.py --mode [basic|conversation]
"""

import argparse
import asyncio
import os
import sys
from dotenv import load_dotenv

# Try importing the google.antigravity modules.
# Since the SDK only publishes pre-compiled wheels for Linux (manylinux) and macOS,
# native Windows environments will fail to import this package.
try:
    from google.antigravity import Agent, LocalAgentConfig, CapabilitiesConfig
    from google.antigravity.utils.interactive import run_interactive_loop
    from google.antigravity.connections.local import LocalConnectionStrategy
    from google.antigravity.conversation.conversation import Conversation
    from google.antigravity.tools.tool_runner import ToolRunner
    from google.antigravity.types import GeminiConfig
    SDK_AVAILABLE = True
except ModuleNotFoundError:
    SDK_AVAILABLE = False


def print_windows_fallback_instructions():
    """Prints guidance on how to run google-antigravity on unsupported Windows environments."""
    print("=" * 80)
    print("[ERROR] GOOGLE ANTIGRAVITY SDK IS NOT INSTALLED OR UNAVAILABLE ON THIS PLATFORM")
    print("=" * 80)
    print("Explanation:")
    print("  The google-antigravity package contains platform-specific precompiled binaries.")
    print("  Currently, official wheels are only built and distributed for:")
    print("    - Linux (manylinux_2_17_x86_64, manylinux_2_17_aarch64)")
    print("    - macOS (macosx_11_0_arm64)")
    print("  Since you are running on native Windows, pip cannot install the matching wheel.")
    print("\nHow to Resolve and Run this Script:")
    print("  1. Windows Subsystem for Linux (WSL):")
    print("     Run this script inside a WSL terminal (e.g. Ubuntu). You can install the package there:")
    print("       pip install google-antigravity")
    print("  2. Docker Container:")
    print("     Run in a python:3.12 Linux container. Mount your workspace, install the SDK, and execute:")
    print("       docker run -it -v ${PWD}:/app -w /app python:3.12 bash")
    print("       pip install google-antigravity")
    print("       python scripts/interactive_loop.py")
    print("  3. Production Deployments (Heroku/Linux):")
    print("     This SDK will work out-of-the-box when running in Heroku or other Linux cloud environments.")
    print("=" * 80)


async def run_basic_loop():
    """Runs the basic interactive loop using Agent and LocalAgentConfig."""
    print("Starting Basic Interactive Loop...")
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
         print("Warning: GEMINI_API_KEY is not set in environment or .env. Using default configuration.")
    
    # Configure the local agent with default capabilities (e.g., terminal/file read-write access)
    config = LocalAgentConfig(
        api_key=api_key,
        capabilities=CapabilitiesConfig(),
    )
    
    # Run the interactive terminal loop. This blocks until the user exits or terminates the session.
    async with Agent(config) as agent:
        await run_interactive_loop(agent)


async def run_conversation_loop():
    """Runs the advanced conversation flow using the Conversation strategy for full lifecycle control."""
    print("Starting Advanced Conversation Loop...")
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
         print("Warning: GEMINI_API_KEY is not set in environment or .env. Using default configuration.")
         
    # Initialize the ToolRunner to handle local tool executions
    tool_runner = ToolRunner()
    
    # Define connection strategy targeting the local machine with tools enabled
    strategy = LocalConnectionStrategy(
        tool_runner=tool_runner,
        gemini_config=GeminiConfig(api_key=api_key) if api_key else None,
    )
    
    # Start a stateful conversational session with step introspection and history tracking
    async with Conversation.create(strategy) as conversation:
        # High-level API: Sends a query, drives any necessary agent/tool steps, and aggregates the result
        print("\n--- Sending High-Level Chat query... ---")
        response = await conversation.chat("What files are here?")
        print("Response from conversation:")
        print(await response.text())
        
        # Accessing session/state metrics
        print("\n--- Session Introspection ---")
        print(f"Total steps taken: {len(conversation.history)}")
        print(f"Turns: {conversation.turn_count}")
        print(f"Last response: {conversation.last_response}")
        
        # Low-level API: Sending a query and iterating over the raw agent streaming steps
        print("\n--- Sending Low-Level Streaming query... ---")
        await conversation.send("Tell me more.")
        
        async for step in conversation.receive_steps():
            # Filter and output the final consolidated agent response
            if step.is_complete_response:
                print(f"Step Result: {step.content}")


def main():
    # Load environment configurations from a local .env file if present
    load_dotenv()

    parser = argparse.ArgumentParser(description="Google Antigravity Interactive Loop Runner")
    parser.add_argument(
        "--mode",
        choices=["basic", "conversation"],
        default="conversation",
        help="Select interactive loop mode (basic or conversation). Defaults to conversation."
    )
    args = parser.parse_args()

    # Gate verification: verify if SDK was imported successfully
    if not SDK_AVAILABLE:
        print_windows_fallback_instructions()
        sys.exit(1)

    # Run selected loop based on mode
    if args.mode == "basic":
        asyncio.run(run_basic_loop())
    else:
        asyncio.run(run_conversation_loop())


if __name__ == "__main__":
    main()
