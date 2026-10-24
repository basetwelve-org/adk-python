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

"""Native Responses conversion; no Responses-to-chat projection or history map."""

from __future__ import annotations

import copy
import json
from typing import Any
from typing import AsyncGenerator
from typing import TYPE_CHECKING

from google.genai import types

from ..tools._custom_tool import _custom_tool_call_part
from ..tools._custom_tool import _custom_tool_input
from .interactions_utils import extract_system_instruction
from .llm_request import LlmRequest
from .llm_response import LlmResponse

if TYPE_CHECKING:
  from .lite_llm import LiteLlm

# The metadata is part of Content, so native event storage and clone/resume
# carry it. It is never displayed, encoded as text, or a Gemini signature.
NATIVE_RESPONSE_KEY = 'adk_native_responses'


def _dump(value: Any) -> dict[str, Any]:
  if isinstance(value, dict):
    return value
  return value.model_dump(mode='json', exclude_none=True)


def _native_response(content: types.Content, model: str) -> dict | None:
  if content.role != 'model':
    return None
  for part in content.parts or []:
    saved = (part.part_metadata or {}).get(NATIVE_RESPONSE_KEY)
    if isinstance(saved, dict) and saved.get('model') == model:
      return saved
  return None


async def _input_items(
    contents: list[types.Content], model: str, conversion: dict[str, Any]
) -> list[dict[str, Any]]:
  from litellm.completion_extras.litellm_responses_transformation.transformation import LiteLLMResponsesTransformationHandler

  from .lite_llm import _content_to_message_param

  bridge = LiteLLMResponsesTransformationHandler()
  custom_ids = set()
  for content in contents:
    saved = _native_response(content, model)
    if saved:
      custom_ids.update(
          item['call_id']
          for item in saved['output']
          if item.get('type') == 'custom_tool_call'
      )
    for part in content.parts or []:
      if part.function_call and _custom_tool_input(part) is not None:
        custom_ids.add(part.function_call.id)

  items = []
  seen_responses = set()
  for content in contents:
    saved = _native_response(content, model)
    if saved:
      # ADK can copy/split one event while pairing asynchronous tool results.
      # Replay the original native array once, preserving its exact order.
      response_id = saved.get('id')
      if response_id is None or response_id not in seen_responses:
        if saved.get('status') == 'completed':
          items.extend(copy.deepcopy(saved['output']))
        seen_responses.add(response_id)
      continue
    message = await _content_to_message_param(content, **conversion)
    messages = (
        message if isinstance(message, list) else [message] if message else []
    )
    converted, _ = bridge.convert_chat_completion_messages_to_responses_api(
        messages
    )
    for item in converted:
      if (
          item.get('type') == 'function_call_output'
          and item.get('call_id') in custom_ids
      ):
        item['type'] = 'custom_tool_call_output'
      items.append(item)
  return items


def _reasoning(options: dict, request: LlmRequest) -> None:
  reasoning = dict(options.pop('reasoning', None) or {})
  effort = options.pop('reasoning_effort', None)
  if effort is not None:
    reasoning['effort'] = effort
  reasoning.setdefault('summary', 'auto')
  thinking = request.config.thinking_config
  if thinking:
    budget = thinking.thinking_budget
    if budget is not None and budget not in (-1, 0):
      raise ValueError(
          'Responses supports reasoning effort, not a token thinking_budget'
      )
    level = thinking.thinking_level
    if (
        level is not None
        and str(getattr(level, 'value', level)) != 'THINKING_LEVEL_UNSPECIFIED'
    ):
      reasoning['effort'] = str(getattr(level, 'value', level)).lower()
    elif budget == 0:
      reasoning['effort'] = 'none'
    if thinking.include_thoughts is False:
      reasoning.pop('summary', None)
    elif thinking.include_thoughts is True:
      reasoning['summary'] = 'auto'
  options['reasoning'] = reasoning


def _tool_choice(request: LlmRequest, tools: list[dict], default: Any) -> Any:
  config = request.config.tool_config
  calling = config.function_calling_config if config else None
  if not tools:
    return None
  if not calling or not calling.allowed_function_names or default == 'none':
    return default
  by_name = {tool.get('name'): tool.get('type') for tool in tools}
  names = calling.allowed_function_names
  if any(name not in by_name for name in names):
    raise ValueError(
        'allowed_function_names must name declared Responses tools'
    )
  return {
      'type': 'allowed_tools',
      'mode': 'required' if default == 'required' else 'auto',
      'tools': [{'type': by_name[name], 'name': name} for name in names],
  }


async def _request_options(
    model: LiteLlm, request: LlmRequest, stream: bool
) -> dict:
  from litellm.completion_extras.litellm_responses_transformation.transformation import LiteLLMResponsesTransformationHandler

  from .lite_llm import _get_completion_inputs
  from .lite_llm import _get_provider_from_model
  from .lite_llm import _get_upload_params
  from .lite_llm import _is_proxied_model

  effective_model = request.model or model.model
  options = dict(model._additional_args)
  if options.get('extra_body') is not None:
    options['extra_body'] = copy.deepcopy(options['extra_body'])
  if 'functions' in options:
    raise ValueError('The Responses route requires tool declarations')
  conversion = {
      'provider': _get_provider_from_model(effective_model),
      'model': effective_model,
      'upload_params': _get_upload_params(effective_model, options),
      'is_proxied': _is_proxied_model(effective_model, options),
      'custom_llm_provider': options.get('custom_llm_provider'),
  }
  # Reuse native declaration/schema conversion without projecting history
  # through ChatCompletion (which collapses consecutive reasoning items).
  _, tools, format, params, choice = await _get_completion_inputs(
      request.model_copy(update={'contents': []}),
      effective_model,
      upload_params=conversion['upload_params'],
      is_proxied=conversion['is_proxied'],
      custom_llm_provider=conversion['custom_llm_provider'],
  )
  bridge = LiteLLMResponsesTransformationHandler()
  tools = bridge._convert_tools_to_responses_format(tools or [])
  options.update(
      model=effective_model,
      input=await _input_items(
          request.contents or [], effective_model, conversion
      ),
      # ChatGPT always streams on the wire. Consume its native iterator even
      # for non-streaming ADK callers; SDK response reconstruction can omit
      # output items when the terminal output array is empty.
      stream=True if effective_model.startswith('chatgpt/') else stream,
  )
  instructions = extract_system_instruction(request.config)
  if instructions:
    options['instructions'] = instructions
  options['tools'] = tools or None
  choice = _tool_choice(request, tools, choice or options.get('tool_choice'))
  if choice is not None:
    options['tool_choice'] = choice
  else:
    options.pop('tool_choice', None)
  _reasoning(options, request)
  if request.service_tier is not None:
    options['service_tier'] = request.service_tier
  options.setdefault('store', False)
  options['include'] = list(
      dict.fromkeys(
          [*(options.get('include') or []), 'reasoning.encrypted_content']
      )
  )
  if 'max_completion_tokens' in options:
    options['max_output_tokens'] = options.pop('max_completion_tokens')
  for key, value in (params or {}).items():
    if key not in ('temperature', 'top_p', 'max_completion_tokens'):
      raise ValueError(f'Unsupported Responses generation parameter: {key}')
    options['max_output_tokens' if key == 'max_completion_tokens' else key] = (
        value
    )
  http = request.config.http_options
  if http:
    options['extra_headers'] = {
        **(options.get('extra_headers') or {}),
        **(http.headers or {}),
    }
    if http.timeout is not None:
      options['timeout'] = http.timeout / 1000
    if http.retry_options and http.retry_options.attempts is not None:
      options['num_retries'] = max(0, http.retry_options.attempts - 1)
    if http.extra_body:
      options['extra_body'] = {
          **options.get('extra_body', {}),
          **copy.deepcopy(http.extra_body),
      }
  body = options.setdefault('extra_body', {})
  if 'include' in body:
    body['include'] = list(
        dict.fromkeys([*body['include'], 'reasoning.encrypted_content'])
    )
  if format:
    native_format = (
        {'type': 'json_schema', **format['json_schema']}
        if format.get('type') == 'json_schema'
        else format
    )
    body['text'] = {**body.get('text', {}), 'format': native_format}
  # ChatGPT's allowlist drops otherwise accepted native parameters. Its native
  # extra_body merge runs after that allowlist; no custom transport is needed.
  if effective_model.startswith('chatgpt/'):
    for key in (
        'max_output_tokens',
        'text',
        'parallel_tool_calls',
        'prompt_cache_key',
    ):
      if key in options:
        value = options.pop(key)
        if key == 'text':
          body[key] = {**value, **body.get(key, {})}
        else:
          body.setdefault(key, value)
  return options


def _response(raw: dict, model: str, include_thoughts: bool) -> LlmResponse:
  output = raw.get('output') or []
  status = raw.get('status', 'completed')
  saved = {
      'id': raw.get('id'),
      'model': model,
      'status': status,
      'output': copy.deepcopy(output),
  }
  parts = []
  if status == 'completed':
    for item in output:
      kind = item.get('type')
      if kind in ('function_call', 'custom_tool_call'):
        if item.get('status') not in (None, 'completed') or not item.get(
            'call_id'
        ):
          raise ValueError('Response contains an incomplete tool call')
        if kind == 'custom_tool_call':
          parts.append(
              _custom_tool_call_part(
                  name=item['name'],
                  input=item['input'],
                  call_id=item['call_id'],
              )
          )
        else:
          args = json.loads(item['arguments'])
          if not isinstance(args, dict):
            raise ValueError('Function arguments must be a JSON object')
          parts.append(
              types.Part(
                  function_call=types.FunctionCall(
                      name=item['name'], args=args, id=item['call_id']
                  )
              )
          )
      elif kind == 'message':
        for content in item.get('content', []):
          if content.get('type') == 'output_text':
            parts.append(types.Part(text=content['text']))
          elif content.get('type') == 'refusal':
            parts.append(types.Part(text=content['refusal']))
      elif kind == 'reasoning' and include_thoughts:
        parts.extend(
            types.Part(text=summary['text'], thought=True)
            for summary in item.get('summary', [])
            if summary.get('text')
        )
  if not parts:
    parts.append(types.Part())
  parts[0].part_metadata = {
      **(parts[0].part_metadata or {}),
      NATIVE_RESPONSE_KEY: saved,
  }
  usage = raw.get('usage') or {}
  input_tokens = usage.get('input_tokens', usage.get('prompt_tokens', 0)) or 0
  output_tokens = (
      usage.get('output_tokens', usage.get('completion_tokens', 0)) or 0
  )
  return LlmResponse(
      content=types.Content(role='model', parts=parts),
      turn_complete=True,
      error_code=None if status == 'completed' else 'INCOMPLETE_RESPONSE',
      error_message=None
      if status == 'completed'
      else f'Responses generation ended with status {status}',
      model_version=raw.get('model'),
      usage_metadata=types.GenerateContentResponseUsageMetadata(
          prompt_token_count=input_tokens,
          candidates_token_count=output_tokens,
          total_token_count=usage.get(
              'total_tokens', input_tokens + output_tokens
          ),
          cached_content_token_count=(
              usage.get('input_tokens_details')
              or usage.get('prompt_tokens_details')
              or {}
          ).get('cached_tokens', 0),
          thoughts_token_count=(
              usage.get('output_tokens_details')
              or usage.get('completion_tokens_details')
              or {}
          ).get('reasoning_tokens', 0),
      ),
  )


async def generate_content(
    model: LiteLlm, request: LlmRequest, *, stream: bool
) -> AsyncGenerator[LlmResponse, None]:
  options = await _request_options(model, request, stream)
  response = await model.llm_client.aresponses(**options)
  if hasattr(response, '__aiter__'):
    terminal = None
    completed_items = {}
    async for event in response:
      event = _dump(event)
      if event.get('type') == 'response.output_item.done':
        completed_items[event['output_index']] = event['item']
      elif event.get('type') == 'response.output_text.delta' and stream:
        yield LlmResponse(
            content=types.Content(
                role='model', parts=[types.Part(text=event['delta'])]
            ),
            partial=True,
        )
      elif event.get('type') in (
          'response.completed',
          'response.incomplete',
          'response.failed',
      ):
        terminal = event['response']
    if terminal is None:
      raise ValueError('Responses stream ended without a terminal response')
    if not terminal.get('output') and completed_items:
      terminal = {
          **terminal,
          'output': [item for _, item in sorted(completed_items.items())],
      }
    response = terminal
  raw = _dump(response)
  thinking = request.config.thinking_config
  yield _response(
      raw, options['model'], bool(thinking and thinking.include_thoughts)
  )
