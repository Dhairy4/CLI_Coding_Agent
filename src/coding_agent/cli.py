import sys
from pathlib import Path
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langgraph.types import Command

from coding_agent.config import config
from coding_agent.graph import build_agent_graph
from coding_agent.state import WorkingMemory


def print_banner():
    print("=" * 65)
    print("  AUTONOMOUS CODING AGENT PLATFORM (PHASE 1 - ACI LOOP) ")
    print(f"  Model: {config.model_name} | Provider: {config.model_provider}")
    print(f"  Workspace: {config.workspace_root}")
    print(f"  Approval Mode: {config.approval_mode}")
    print("=" * 65)


def run_cli():
    print_banner()
    agent = build_agent_graph()
    thread_id = "cli-session-1"
    graph_config = {"configurable": {"thread_id": thread_id}}

    working_memory = WorkingMemory()
    initial_state = {
        "messages": [],
        "workspace_root": str(config.workspace_root),
        "repo_snapshot": "",
        "approval_mode": config.approval_mode,
        "working_memory": working_memory,
        "step_count": 0,
        "max_steps": config.max_steps,
        "pending_approval": None,
        "last_command_output": None,
    }

    while True:
        try:
            user_input = input("\n\033[1;36mUser > \033[0m").strip()
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit", "q"):
                print("Exiting coding agent session. Goodbye!")
                break

            working_memory.current_goal = user_input
            stream_input = {"messages": [HumanMessage(content=user_input)]}

            while True:
                # Execute stream
                interrupted = False
                interrupt_data = None

                for event in agent.stream(stream_input, graph_config, stream_mode="updates"):
                    # Check for interrupts
                    if "__interrupt__" in event:
                        interrupted = True
                        interrupt_data = event["__interrupt__"][0].value
                        break

                    for node_name, updates in event.items():
                        messages = updates.get("messages", [])
                        for msg in messages:
                            if isinstance(msg, AIMessage):
                                if msg.content:
                                    print(f"\n\033[1;32m[Agent]\033[0m {msg.content}")
                                if msg.tool_calls:
                                    for tc in msg.tool_calls:
                                        print(
                                            f"\n\033[1;33m[Tool Call]\033[0m {tc['name']}({tc['args']})"
                                        )
                            elif isinstance(msg, ToolMessage):
                                preview = msg.content[:300]
                                if len(msg.content) > 300:
                                    preview += " ... [truncated]"
                                print(f"\033[0;37m[Tool Result]\033[0m {preview}")

                if interrupted and interrupt_data:
                    calls = interrupt_data.get("calls", [])
                    print("\n" + "!" * 65)
                    print("\033[1;31m[APPROVAL REQUIRED]\033[0m")
                    for c in calls:
                        print(f" Tool: \033[1m{c['name']}\033[0m")
                        print(f" Arguments: {c['args']}")
                    print("!" * 65)

                    ans = input("Approve this action? [y/N]: ").strip().lower()
                    approved = ans in ("y", "yes")

                    # Resume graph with user decision
                    stream_input = Command(resume={"approved": approved})
                else:
                    break

        except KeyboardInterrupt:
            print("\nInterrupt received. Session paused.")
            break
        except Exception as e:
            print(f"\n\033[1;31mError:\033[0m {e}")


if __name__ == "__main__":
    run_cli()
