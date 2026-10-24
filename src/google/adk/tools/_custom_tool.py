# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from typing import Any
from typing import TYPE_CHECKING

from google.genai import types

from ._custom_tool_declaration import CustomToolDeclaration
from ._custom_tool_declaration import CustomToolInputFormat
from ._custom_tool_declaration import CustomToolTextFormat
from .base_tool import BaseTool

if TYPE_CHECKING:
  from .tool_context import ToolContext


_CUSTOM_TOOL_TYPE = 'adk_tool_type'


def _custom_tool_call_part(
    *, name: str, input: str, call_id: str | None, will_continue: bool = False
) -> types.Part:
  """Adapt raw input to ADK invocation while preserving its native wire type."""
  return types.Part(
      function_call=types.FunctionCall(
          id=call_id,
          name=name,
          args={'input': input},
          will_continue=True if will_continue else None,
      ),
      part_metadata={_CUSTOM_TOOL_TYPE: 'custom'},
  )


def _custom_tool_input(part: types.Part) -> str | None:
  if (part.part_metadata or {}).get(_CUSTOM_TOOL_TYPE) != 'custom':
    return None
  call = part.function_call
  value = (call.args or {}).get('input') if call else None
  if not isinstance(value, str):
    raise ValueError('Custom tool calls require a raw string input.')
  return value


class CustomTool(BaseTool):
  """A client-executed tool with raw text input, optionally grammar constrained.

  Subclasses implement run_custom_async. The normal ADK tool lifecycle,
  callbacks and events carry the string as args['input']; model adapters send
  native custom declarations and raw input, never a function JSON schema.
  Currently supported through LiteLlm on native OpenAI-compatible providers.
  """

  def __init__(
      self,
      *,
      name: str,
      description: str,
      format: CustomToolInputFormat | None = None,
  ) -> None:
    super().__init__(name=name, description=description)
    self._custom_declaration = CustomToolDeclaration(
        name=name,
        description=description,
        format=format if format is not None else CustomToolTextFormat(),
    )

  def _get_custom_declaration(self) -> CustomToolDeclaration:
    return CustomToolDeclaration(
        name=self.name,
        description=self.description,
        format=self._custom_declaration.format,
    )

  async def run_async(
      self, *, args: dict[str, Any], tool_context: ToolContext
  ) -> Any:
    value = args.get('input')
    if set(args) != {'input'} or not isinstance(value, str):
      raise ValueError('Custom tools take exactly one raw input string.')
    return await self.run_custom_async(input=value, tool_context=tool_context)

  async def run_custom_async(
      self, *, input: str, tool_context: ToolContext
  ) -> Any:
    """Execute the model's unchanged raw input string."""
    raise NotImplementedError(f'{type(self)} is not implemented')
