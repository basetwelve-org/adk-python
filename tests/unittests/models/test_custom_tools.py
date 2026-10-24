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


from google.adk.agents import LlmAgent
from google.adk.agents.run_config import RunConfig
from google.adk.agents.run_config import StreamingMode
from google.adk.models._capabilities import LlmCapabilities
from google.adk.models.base_llm import BaseLlm
from google.adk.models.lite_llm import _content_to_message_param
from google.adk.models.lite_llm import _get_completion_inputs
from google.adk.models.lite_llm import _model_response_to_generate_content_response
from google.adk.models.lite_llm import LiteLlm
from google.adk.models.lite_llm import LiteLLMClient
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import CustomTool
from google.adk.tools import CustomToolGrammarFormat
from google.adk.tools import FunctionTool
from google.genai import types
import litellm
from litellm import ModelResponse
from litellm.types.utils import ModelResponseStream
from pydantic import ValidationError
import pytest

RAW = (
    '*** Begin Patch\n*** Add File: résumé.py\n+print("{not JSON}")\n*** End'
    ' Patch\n'
)
GRAMMAR = CustomToolGrammarFormat(syntax='lark', definition='start: /.+/')


@pytest.fixture(autouse=True)
def provider_metadata(monkeypatch):
  # Resolving chatgpt metadata otherwise starts device authorization, even
  # when the completion client is mocked. Unit tests never authenticate.
  monkeypatch.setattr(
      litellm,
      'get_llm_provider',
      lambda *args, **kwargs: ('gpt-6-luna', 'chatgpt', None, None),
  )
  monkeypatch.setattr(
      litellm, 'supports_response_schema', lambda **kwargs: False
  )


class Recorder(CustomTool):

  def __init__(self):
    super().__init__(
        name='apply_patch', description='Raw patch', format=GRAMMAR
    )
    self.inputs = []

  async def run_custom_async(self, *, input, tool_context):
    self.inputs.append(input)
    return {'applied': True}


def response(calls, finish='tool_calls'):
  return ModelResponse(
      model='gpt-6-luna',
      choices=[{
          'index': 0,
          'finish_reason': finish,
          'message': {
              'role': 'assistant',
              'content': None,
              'tool_calls': calls,
          },
      }],
  )


def custom_call(raw=RAW):
  return {
      'type': 'custom',
      'id': 'call_patch',
      'custom': {'name': 'apply_patch', 'input': raw},
  }


@pytest.mark.asyncio
async def test_custom_declaration_invocation_and_history_are_native():
  tool = Recorder()
  request = LlmRequest()
  request.append_tools([FunctionTool(lambda value: value), tool])
  assert request.tools_dict['apply_patch'] is tool
  assert len(request.config.tools[0].function_declarations) == 1
  messages, tools, *_ = await _get_completion_inputs(
      request, 'chatgpt/gpt-6-luna'
  )
  declared = next(item for item in tools if item['type'] == 'custom')
  assert declared == {
      'type': 'custom',
      'name': 'apply_patch',
      'description': 'Raw patch',
      'format': GRAMMAR.model_dump(),
  }
  assert 'parameters' not in declared
  assert tool._get_declaration() is None
  assert await tool.run_async(args={'input': RAW}, tool_context=None) == {
      'applied': True
  }
  assert tool.inputs == [RAW]
  with pytest.raises(ValueError):
    await tool.run_async(args={'input': {'wrapped': 'JSON'}}, tool_context=None)
  with pytest.raises(ValidationError):
    CustomToolGrammarFormat(syntax='unknown', definition='bad')

  converted = _model_response_to_generate_content_response(
      response([custom_call()])
  )
  # Session serialization must preserve the type even after this tool is filtered out.
  persisted = types.Content.model_validate_json(
      converted.content.model_dump_json()
  )
  message = await _content_to_message_param(persisted)
  assert message['tool_calls'] == [custom_call()]
  assert persisted.parts[0].function_call.args == {'input': RAW}


@pytest.mark.parametrize('raw', [RAW, '', '{"input":"this is raw text"}'])
def test_nonstream_custom_input_is_never_json_parsed(raw):
  native = _model_response_to_generate_content_response(
      response([custom_call(raw)])
  )
  assert native.content.parts[0].function_call.args == {'input': raw}
  # LiteLLM versions may surface native Responses input in function.arguments.
  normalized = _model_response_to_generate_content_response(
      response([{
          'type': 'function',
          'id': 'call_patch',
          'function': {'name': 'apply_patch', 'arguments': raw},
      }]),
      custom_tool_names=frozenset({'apply_patch'}),
  )
  assert normalized.content == native.content
  truncated = _model_response_to_generate_content_response(
      response([custom_call(raw)], finish='length')
  )
  assert truncated.error_code == types.FinishReason.MAX_TOKENS
  assert not truncated.content.parts


class Client(LiteLLMClient):

  def __init__(self, finish='tool_calls', normalized=False):
    self.calls = []
    self.finish = finish
    self.normalized = normalized

  async def acompletion(self, **kwargs):
    self.calls.append(kwargs)
    if len(self.calls) > 1:
      if not kwargs.get('stream'):
        return ModelResponse(
            model='gpt-6-luna',
            choices=[{
                'index': 0,
                'finish_reason': 'stop',
                'message': {'role': 'assistant', 'content': 'Done'},
            }],
        )

      async def final():
        yield ModelResponseStream(
            model='gpt-6-luna',
            choices=[{
                'index': 0,
                'finish_reason': 'stop',
                'delta': {'content': 'Done'},
            }],
        )

      return final()
    if not kwargs.get('stream'):
      return response(
          [
              custom_call(),
              {
                  'type': 'function',
                  'id': 'call_lookup',
                  'function': {'name': 'lookup', 'arguments': '{"value":42}'},
              },
          ],
          self.finish,
      )
    # Interleaved calls, braces/newlines and escaped quotes in raw input must
    # not trigger JSON completion tracking or alter call order/identity.
    chunks = [
        {
            'index': 0,
            'id': 'call_patch',
            'type': 'custom',
            'custom': {'name': 'apply_patch', 'input': RAW[:20]},
        },
        {
            'index': 1,
            'id': 'call_lookup',
            'type': 'function',
            'function': {'name': 'lookup', 'arguments': '{"value":'},
        },
        {'index': 0, 'type': 'custom', 'custom': {'input': RAW[20:]}},
        {'index': 1, 'type': 'function', 'function': {'arguments': '42}'}},
    ]

    async def stream():
      for chunk in chunks:
        if self.normalized and chunk.get('type') == 'custom':
          payload = chunk.pop('custom')
          chunk['type'] = 'function'
          chunk['function'] = {'arguments': payload['input']}
          if payload.get('name'):
            chunk['function']['name'] = payload['name']
        yield ModelResponseStream(
            model='gpt-6-luna',
            choices=[{
                'index': 0,
                'delta': {'tool_calls': [chunk]},
                'finish_reason': None,
            }],
        )
      yield ModelResponseStream(
          model='gpt-6-luna',
          choices=[{'index': 0, 'delta': {}, 'finish_reason': self.finish}],
      )

    return stream()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'stream,normalized', [(False, False), (True, False), (True, True)]
)
async def test_runner_dispatches_raw_custom_and_function_once_and_replays(
    stream, normalized, monkeypatch
):
  tool = Recorder()
  values = []

  def lookup(value: int):
    values.append(value)
    return value

  client = Client(normalized=normalized)
  model = LiteLlm(model='chatgpt/gpt-6-luna')
  model.llm_client = client
  monkeypatch.setattr(LiteLLMClient, 'acompletion', client.acompletion)
  sessions = InMemorySessionService()
  await sessions.create_session(
      app_name='customtools', user_id='user', session_id='one'
  )
  agent = LlmAgent(name='test', model=model, tools=[tool, FunctionTool(lookup)])
  runner = Runner(agent=agent, app_name='customtools', session_service=sessions)
  events = [
      event
      async for event in runner.run_async(
          user_id='user',
          session_id='one',
          new_message=types.Content(
              role='user', parts=[types.Part(text='Patch and look up')]
          ),
          run_config=RunConfig(
              streaming_mode=StreamingMode.SSE
              if stream
              else StreamingMode.NONE,
              max_llm_calls=3,
          ),
      )
  ]
  assert tool.inputs == [RAW]
  assert values == [42]
  assert len(client.calls) == 2
  calls = [
      call
      for message in client.calls[1]['messages']
      for call in message.get('tool_calls', [])
  ]
  assert calls[0] == custom_call()
  assert calls[1]['function'] == {
      'name': 'lookup',
      'arguments': '{"value": 42}',
  }
  assert client.calls[0]['tools'][-1]['type'] == 'custom'
  session = await sessions.get_session(
      app_name='customtools', user_id='user', session_id='one'
  )
  final_calls = [
      part.function_call
      for event in session.events
      for part in (event.content.parts or [] if event.content else [])
      if part.function_call
  ]
  assert [call.id for call in final_calls] == ['call_patch', 'call_lookup']


@pytest.mark.asyncio
async def test_truncated_stream_is_not_dispatched():
  request = LlmRequest()
  request.append_tools([Recorder()])
  model = LiteLlm(model='chatgpt/gpt-6-luna')
  model.llm_client = Client(finish='length')
  results = [
      item async for item in model.generate_content_async(request, stream=True)
  ]
  complete = [item for item in results if not item.partial]
  assert len(complete) == 1
  assert complete[0].error_code == types.FinishReason.MAX_TOKENS
  assert not complete[0].content


@pytest.mark.asyncio
async def test_unsupported_adapter_refuses_instead_of_dropping_custom_tools():
  class Unsupported(BaseLlm):
    called: bool = False

    @property
    def capabilities(self):
      return LlmCapabilities()

    async def generate_content_async(self, request, stream=False):
      self.called = True
      yield LlmResponse(
          content=types.Content(role='model', parts=[types.Part(text='bad')])
      )

  model = Unsupported(model='unsupported')
  sessions = InMemorySessionService()
  await sessions.create_session(
      app_name='customtools', user_id='user', session_id='one'
  )
  runner = Runner(
      agent=LlmAgent(name='test', model=model, tools=[Recorder()]),
      app_name='customtools',
      session_service=sessions,
  )
  with pytest.raises(ValueError, match='does not support native custom tools'):
    async for _ in runner.run_async(
        user_id='user',
        session_id='one',
        new_message=types.Content(
            role='user', parts=[types.Part(text='Patch')]
        ),
    ):
      pass
  assert not model.called


@pytest.mark.asyncio
async def test_litellm_chatgpt_boundary_preserves_native_grammar_and_history():
  from litellm.completion_extras.litellm_responses_transformation.transformation import LiteLLMResponsesTransformationHandler

  request = LlmRequest()
  request.append_tools([Recorder()])
  _, tools, *_ = await _get_completion_inputs(request, 'chatgpt/gpt-6-luna')
  bridge = LiteLLMResponsesTransformationHandler()
  native = bridge._convert_tools_to_responses_format(tools)
  assert native == tools
  assert native[0]['type'] == 'custom'
  assert native[0]['format'] == GRAMMAR.model_dump()


@pytest.mark.asyncio
async def test_toolset_prefix_keeps_custom_declaration_and_dispatch_name_together():
  from google.adk.tools.base_toolset import BaseToolset

  original = Recorder()

  class Toolset(BaseToolset):

    async def get_tools(self, readonly_context=None):
      return [original]

  tools = await Toolset(tool_name_prefix='workspace').get_tools_with_prefix()
  request = LlmRequest()
  request.append_tools(tools)
  assert original.name == 'apply_patch'
  assert list(request.tools_dict) == ['workspace_apply_patch']
  assert request.custom_tool_declarations[0].name == 'workspace_apply_patch'
  assert await request.tools_dict['workspace_apply_patch'].run_async(
      args={'input': RAW}, tool_context=None
  ) == {'applied': True}
  assert original.inputs == [RAW]
