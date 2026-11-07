"""Tool schemas shared by all agents (provider-neutral JSON-schema form)."""


def bash_tool(timeout: float) -> dict:
    return {
        "name": "bash",
        "description": (
            "Run a bash command in the workspace (your current directory). Returns the combined stdout/stderr "
            f"(truncated if long) and the exit code. Each command is limited to {int(timeout)} seconds."
        ),
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string", "description": "The bash command to run."}},
            "required": ["command"],
        },
    }


WRITE_FILE = {
    "name": "write_file",
    "description": "Create or overwrite a text file in the workspace. Paths are relative to the workspace root.",
    "parameters": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Relative path, e.g. solution.py"},
            "content": {"type": "string", "description": "Full new file content."},
        },
        "required": ["path", "content"],
    },
}

SUBMIT = {
    "name": "submit",
    "description": "Finish the task and submit the current state of the workspace for grading.",
    "parameters": {
        "type": "object",
        "properties": {"summary": {"type": "string", "description": "One or two sentences on what you did."}},
        "required": ["summary"],
    },
}

ABSTAIN = {
    "name": "abstain",
    "description": (
        "End the task without submitting, because it cannot be completed correctly within your remaining budget "
        "or cannot be solved as specified."
    ),
    "parameters": {
        "type": "object",
        "properties": {"reason": {"type": "string", "description": "Short explanation."}},
        "required": ["reason"],
    },
}

TERMINAL = {"submit", "abstain"}


def tool_set(cmd_timeout: float, abstain: bool) -> list[dict]:
    tools = [bash_tool(cmd_timeout), WRITE_FILE, SUBMIT]
    if abstain:
        tools.append(ABSTAIN)
    return tools
