# Native custom tools

`CustomTool` accepts one raw string, optionally constrained by a provider-native Lark or regex grammar.
Use it for freeform input such as patches; keep ordinary structured arguments in `FunctionTool`.
The declaration is sent as `type: "custom"` with `name`, `description` and `format`, with no function
JSON schema or JSON-wrapping instruction.

```python
from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools import CustomTool, CustomToolGrammarFormat

class RememberExpression(CustomTool):
    def __init__(self):
        super().__init__(
            name="remember_expression",
            description="Remember an addition expression exactly as written.",
            format=CustomToolGrammarFormat(
                syntax="lark",
                definition='start: INT " + " INT\n%import common.INT',
            ),
        )

    async def run_custom_async(self, *, input, tool_context):
        tool_context.state["expression"] = input
        return {"remembered": True}

agent = LlmAgent(
    name="expressions",
    model=LiteLlm(model="openai/gpt-6-luna"),
    tools=[RememberExpression()],
)
```

Omit `format` for unconstrained text, or use `CustomToolTextFormat`. `CustomToolGrammarFormat` takes
`syntax="lark"` or `syntax="regex"` and a nonempty `definition`. The provider validates grammar support;
there is no conversion to a function-tool fallback. Unsupported ADK model adapters raise before calling
the model instead of silently dropping the declaration. The LiteLlm implementation supports native
OpenAI-compatible providers (`openai`, `azure`, `chatgpt`), tested with LiteLLM 1.105.1. The selected
backend/model must itself accept custom tools.

Streaming fragments are assembled as raw text, independently from function JSON arguments. A truncated
custom call is not dispatched. Normal ADK execution, tool context and callbacks remain in place:
`run_custom_async` receives the unchanged string, while the shared callback/event representation carries
`args={"input": raw_string}`. That internal mapping is not the model's wire format. Part metadata records
the custom call type, so persisted history is replayed as a native custom call, including when that tool
is no longer offered in a later turn. Function tool declarations, parsing and history stay unchanged.
