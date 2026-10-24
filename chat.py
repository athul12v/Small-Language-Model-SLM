"""
chat.py - Simple Terminal CLI for SLM Enterprise Assistant.
"""

from harness import Harness


def main():
    print("=" * 60)
    print("SLM ENTERPRISE ASSISTANT (CLI)")
    print("=" * 60)
    print("Guarded by the AI Harness. Type 'exit' to quit.\n")

    harness = Harness()
    session_id = "cli_session"

    while True:
        try:
            query = input("Employee: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not query:
            continue
        if query.lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        # Process through harness
        res = harness.process(query=query, session_id=session_id)

        print(f"\n[Status: {res['status']}]")

        if res.get("answer"):
            print(f"Answer: {res['answer']}")

        if res.get("follow_up_question"):
            print(f"Clarification needed: {res['follow_up_question']}")

        if res.get("sources"):
            print(f"Supporting Document: {res['sources'][0].get('instruction', '')}")

        print("-" * 40 + "\n")


if __name__ == "__main__":
    main()