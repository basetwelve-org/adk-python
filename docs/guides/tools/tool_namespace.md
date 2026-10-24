# Native tool namespaces and deferred loading

`ToolNamespace` groups related functions for the native Responses API. Import it
from `google.adk.tools`, give it a name and a short purpose description, and pass
it to `FunctionTool(..., namespace=namespace, defer_loading=True)`. Custom
`BaseTool` subclasses accept the same fields. A `BaseToolset` can set them for
all returned tools; metadata is applied to copies, preserving shared instances.

Use `LiteLlm(..., use_responses_api=True)` with a provider/model that supports
native tool search. Deferred declarations automatically add hosted `tool_search`.
The adapter rejects namespaces on its Chat Completions route. It does not flatten
or eagerly expose deferred declarations as a compatibility fallback.

Tools dispatch internally by `namespace.name`, while the wire keeps namespace
and name separate. Native search calls, search outputs, reasoning and function
calls remain in their original order in persisted event metadata, including
namespace, and are replayed unchanged after JSON storage or session restart.
Streaming uses completed output items when the terminal output array is empty.

With deferred search, `tool_choice` stays `auto`, preserving the declarations.
`allowed_function_names` is enforced by the executor, with qualified names for
namespace-specific restrictions. A disallowed call returns an error with code
`TOOL_NOT_ALLOWED_THIS_TURN` and lets the model continue. Function calling mode
NONE similarly refuses execution. An eager-only request still uses native
Responses `allowed_tools`. Authorization remains the application's responsibility.

Keep searchable namespace descriptions short and specific. Prefer fewer than
ten related functions per namespace, and keep parameter and output detail in
the deferred function descriptions.
