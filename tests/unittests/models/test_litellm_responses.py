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

"""Native Responses history is durable data, not a lossy chat projection."""

import copy
import json

from google.adk.agents.llm_agent import LlmAgent
from google.adk.apps.app import App
from google.adk.apps.app import EventsCompactionConfig
from google.adk.apps.app import ResumabilityConfig
from google.adk.apps.llm_event_summarizer import LlmEventSummarizer
from google.adk.events.event import Event
from google.adk.events.event_actions import EventActions
from google.adk.events.event_actions import EventCompaction
from google.adk.flows.llm_flows.context._contents import _get_contents
from google.adk.models._lite_llm_responses import _response
from google.adk.models._lite_llm_responses import NATIVE_RESPONSE_KEY
from google.adk.models.lite_llm import LiteLlm
from google.adk.models.lite_llm import LiteLLMClient
from google.adk.models.llm_request import LlmRequest
from google.adk.runners import Runner
from google.adk.sessions.database_session_service import DatabaseSessionService
from google.adk.tools import CustomTool
from google.adk.tools import CustomToolGrammarFormat
from google.adk.tools.function_tool import FunctionTool
from google.genai import types
import litellm
import pytest

MODEL = 'openai/gpt-5'
RAW = '*** Begin Patch\n*** Add File: café.py\n+print("{raw}")\n*** End Patch\n'


@pytest.fixture(autouse=True)
def provider_metadata(monkeypatch):
  # No unit test authenticates or calls a remote model.
  monkeypatch.setattr(
      litellm,
      'get_llm_provider',
      lambda *a, **kw: ('gpt-5', 'openai', None, None),
  )
  monkeypatch.setattr(litellm, 'supports_response_schema', lambda **kw: True)


def reasoning(id):
  return {
      'type': 'reasoning',
      'id': id,
      'encrypted_content': f'opaque-{id}',
      'summary': [{'type': 'summary_text', 'text': f'Summary {id}'}],
  }


def message(id='message', text='Done.', phase='final_answer'):
  return {
      'type': 'message',
      'id': id,
      'role': 'assistant',
      'status': 'completed',
      'phase': phase,
      'content': [{'type': 'output_text', 'text': text, 'annotations': []}],
  }


def call(name='lookup', id='call_lookup', arguments='{"key":"alpha"}'):
  return {
      'type': 'function_call',
      'id': f'fc_{id}',
      'call_id': id,
      'name': name,
      'arguments': arguments,
      'status': 'completed',
  }


def response(output, id='response', status='completed'):
  return {
      'id': id,
      'object': 'response',
      'model': 'gpt-5',
      'status': status,
      'output': output,
      'usage': {
          'input_tokens': 100,
          'output_tokens': 20,
          'total_tokens': 120,
          'input_tokens_details': {'cached_tokens': 64},
          'output_tokens_details': {'reasoning_tokens': 12},
      },
  }


class Client(LiteLLMClient):

  def __init__(self, responses):
    self.responses = list(responses)
    self.requests = []

  async def aresponses(self, **kwargs):
    self.requests.append(copy.deepcopy(kwargs))
    result = self.responses.pop(0)
    return result() if callable(result) else result


def lookup(key: str) -> dict:
  """Read a synthetic fixture by key."""
  return {'key': key, 'value': 42}


class Patch(CustomTool):

  def __init__(self):
    super().__init__(
        name='apply_patch',
        description='Apply raw patch',
        format=CustomToolGrammarFormat(syntax='lark', definition='start: /.+/'),
    )

  async def run_custom_async(self, *, input, tool_context):
    return {'applied': input}


def request(contents=None, model=MODEL):
  result = LlmRequest(
      model=model,
      contents=contents
      or [
          types.Content(role='user', parts=[types.Part(text='Read and patch.')])
      ],
      config=types.GenerateContentConfig(
          system_instruction='Stable application policy.'
      ),
  )
  result.append_tools([FunctionTool(lookup), Patch()])
  return result


@pytest.mark.parametrize('stream', [False, True])
@pytest.mark.asyncio
async def test_all_native_outputs_survive_json_and_a_fresh_model_in_order(
    stream,
):
  outputs = [
      reasoning('r1'),
      reasoning('r2'),
      reasoning('r3'),
      message('commentary', 'Checking.', 'commentary'),
      call(),
      {
          'type': 'custom_tool_call',
          'id': 'ct_patch',
          'call_id': 'call_patch',
          'name': 'apply_patch',
          'input': RAW,
          'status': 'completed',
      },
  ]
  native = response(outputs)

  async def chunks():
    yield {'type': 'response.output_text.delta', 'delta': 'Checking.'}
    yield {'type': 'response.completed', 'response': native}

  client = Client([chunks if stream else native])
  model = LiteLlm(
      model=MODEL,
      use_responses_api=True,
      llm_client=client,
      reasoning_effort='medium',
  )
  generated = [
      r async for r in model.generate_content_async(request(), stream=stream)
  ]
  final = generated[-1]
  assert [r.partial for r in generated[:-1]] == ([True] if stream else [])
  assert final.usage_metadata.cached_content_token_count == 64
  assert final.usage_metadata.thoughts_token_count == 12
  assert all(
      not part.thought_signature and 'opaque-' not in (part.text or '')
      for part in final.content.parts
  )
  restored = Event.model_validate_json(
      Event(author='agent', content=final.content).model_dump_json()
  )
  stored = restored.content.parts[0].part_metadata[NATIVE_RESPONSE_KEY]
  assert stored['output'] == outputs
  assert [t['name'] for t in client.requests[0]['tools']] == [
      'lookup',
      'apply_patch',
  ]
  assert client.requests[0]['tools'][1]['format']['type'] == 'grammar'
  assert client.requests[0]['include'] == ['reasoning.encrypted_content']
  assert client.requests[0]['reasoning'] == {
      'effort': 'medium',
      'summary': 'auto',
  }
  assert client.requests[0]['store'] is False
  assert client.requests[0]['instructions'] == 'Stable application policy.'
  assert 'messages' not in client.requests[0]

  fresh = Client([response([message()], 'next-response')])
  continuation = request([
      *request().contents,
      restored.content,
      types.Content(
          role='user',
          parts=[
              types.Part(
                  function_response=types.FunctionResponse(
                      name='lookup', id='call_lookup', response={'value': 42}
                  )
              ),
              types.Part(
                  function_response=types.FunctionResponse(
                      name='apply_patch',
                      id='call_patch',
                      response={'applied': True},
                  )
              ),
          ],
      ),
  ])
  new_model = LiteLlm(model=MODEL, use_responses_api=True, llm_client=fresh)
  _ = [r async for r in new_model.generate_content_async(continuation)]
  items = fresh.requests[0]['input']
  assert items[1 : 1 + len(outputs)] == outputs
  assert [
      (i['type'], i['call_id'])
      for i in items
      if i['type'].endswith('call_output')
  ] == [
      ('function_call_output', 'call_lookup'),
      ('custom_tool_call_output', 'call_patch'),
  ]
  assert (
      restored.content.parts[0].part_metadata[NATIVE_RESPONSE_KEY]['output']
      == outputs
  )


@pytest.mark.asyncio
async def test_native_config_maps_effort_summary_selection_schema_and_http_without_shared_mutation():
  original = {'text': {'verbosity': 'high'}, 'kept': {'a': 1}}
  client = Client([response([message()]), response([message()], 'second')])
  model = LiteLlm(
      model='chatgpt/gpt-6-luna',
      use_responses_api=True,
      llm_client=client,
      reasoning_effort='low',
      extra_body=original,
      extra_headers={'existing': 'kept'},
      timeout=90,
  )
  req = request(model=model.model)
  req.service_tier = 'priority'
  req.config.thinking_config = types.ThinkingConfig(
      thinking_level='HIGH', include_thoughts=True
  )
  req.config.max_output_tokens = 77
  req.config.response_schema = {
      'type': 'object',
      'properties': {'answer': {'type': 'string'}},
      'required': ['answer'],
  }
  req.config.tool_config = types.ToolConfig(
      function_calling_config=types.FunctionCallingConfig(
          mode='ANY', allowed_function_names=['lookup']
      )
  )
  req.config.http_options = types.HttpOptions(
      timeout=12345,
      headers={'session_id': 'stable'},
      retry_options=types.HttpRetryOptions(attempts=3),
  )
  _ = [r async for r in model.generate_content_async(req)]
  sent = client.requests[0]
  assert sent['reasoning'] == {'effort': 'high', 'summary': 'auto'}
  assert sent['service_tier'] == 'priority'
  assert sent['tool_choice'] == {
      'type': 'allowed_tools',
      'mode': 'required',
      'tools': [{'type': 'function', 'name': 'lookup'}],
  }
  assert sent['extra_body']['max_output_tokens'] == 77
  assert sent['extra_body']['text']['format']['strict'] is True
  assert sent['extra_body']['text']['verbosity'] == 'high'
  assert sent['extra_headers'] == {'existing': 'kept', 'session_id': 'stable'}
  assert sent['timeout'] == 12.345 and sent['num_retries'] == 2
  assert original == {'text': {'verbosity': 'high'}, 'kept': {'a': 1}}
  req.config.thinking_config = types.ThinkingConfig(
      thinking_budget=0, include_thoughts=False
  )
  req.config.tool_config.function_calling_config.allowed_function_names = [
      'apply_patch'
  ]
  _ = [r async for r in model.generate_content_async(req)]
  assert client.requests[1]['reasoning'] == {'effort': 'none'}
  assert client.requests[1]['tool_choice']['tools'] == [
      {'type': 'custom', 'name': 'apply_patch'}
  ]
  assert client.requests[0]['tools'] == client.requests[1]['tools']
  req.config.thinking_config = types.ThinkingConfig(thinking_budget=256)
  with pytest.raises(ValueError, match='thinking_budget'):
    _ = [r async for r in model.generate_content_async(req)]
  assert len(client.requests) == 2


@pytest.mark.asyncio
async def test_incomplete_stream_is_durable_but_never_executable_or_replayed():
  incomplete = response(
      [
          reasoning('partial'),
          {**call(), 'arguments': '{"key":', 'status': 'in_progress'},
      ],
      status='incomplete',
  )

  async def chunks():
    yield {'type': 'response.output_text.delta', 'delta': 'Working'}
    yield {'type': 'response.function_call_arguments.delta', 'delta': '{"key":'}
    yield {'type': 'response.incomplete', 'response': incomplete}

  client = Client([chunks, response([message()], 'second')])
  model = LiteLlm(model=MODEL, use_responses_api=True, llm_client=client)
  result = [
      r async for r in model.generate_content_async(request(), stream=True)
  ]
  assert result[0].partial and result[0].content.parts[0].part_metadata is None
  assert result[-1].error_code == 'INCOMPLETE_RESPONSE'
  assert not result[-1].get_function_calls()
  assert (
      result[-1].content.parts[0].part_metadata[NATIVE_RESPONSE_KEY]['output']
      == incomplete['output']
  )
  continuation = request([
      result[-1].content,
      types.Content(role='user', parts=[types.Part(text='Try again.')]),
  ])
  _ = [r async for r in model.generate_content_async(continuation)]
  assert not any(
      i['type'] in ('function_call', 'reasoning')
      for i in client.requests[-1]['input']
  )


@pytest.mark.asyncio
async def test_sqlite_tool_continuation_cross_turn_and_agent_clone_need_no_live_model_state(
    tmp_path,
):
  raw = [
      reasoning('one'),
      reasoning('two'),
      message('note', 'Read it.', 'commentary'),
      call(),
  ]
  first = Client([response(raw), response([message()], 'answer')])
  model = LiteLlm(model=MODEL, use_responses_api=True, llm_client=first)
  root = LlmAgent(
      name='agent', model=model, instruction='Read the fixture.', tools=[lookup]
  )
  url = f'sqlite+aiosqlite:///{tmp_path / "events.db"}'
  sessions = DatabaseSessionService(db_url=url)
  session = await sessions.create_session(
      app_name='native', user_id='user', session_id='chat'
  )
  runner = Runner(
      app=App(
          name='native',
          root_agent=root,
          resumability_config=ResumabilityConfig(is_resumable=True),
      ),
      session_service=sessions,
  )
  events = [
      e
      async for e in runner.run_async(
          user_id='user',
          session_id=session.id,
          new_message=types.Content(
              role='user', parts=[types.Part(text='Read alpha.')]
          ),
      )
  ]
  assert any(e.get_function_responses() for e in events)
  assert [
      i
      for i in first.requests[1]['input']
      if i.get('id') in {x['id'] for x in raw}
  ] == raw
  assert any(
      i.get('type') == 'function_call_output' and i['call_id'] == 'call_lookup'
      for i in first.requests[1]['input']
  )
  await runner.close()
  await sessions.close()

  fresh = Client([
      response([message('followup', '42.')], 'third'),
      response([message('isolated')], 'isolated'),
  ])
  restarted = root.clone(
      update={
          'model': LiteLlm(
              model=MODEL, use_responses_api=True, llm_client=fresh
          )
      }
  )
  sessions = DatabaseSessionService(db_url=url)
  stored = await sessions.get_session(
      app_name='native', user_id='user', session_id='chat'
  )
  saved = [
      p.part_metadata[NATIVE_RESPONSE_KEY]
      for e in stored.events
      if e.content
      for p in e.content.parts or []
      if NATIVE_RESPONSE_KEY in (p.part_metadata or {})
  ]
  assert saved[0]['output'] == raw
  runner = Runner(
      app=App(
          name='native',
          root_agent=restarted,
          resumability_config=ResumabilityConfig(is_resumable=True),
      ),
      session_service=sessions,
  )
  _ = [
      e
      async for e in runner.run_async(
          user_id='user',
          session_id='chat',
          new_message=types.Content(
              role='user', parts=[types.Part(text='What was its value?')]
          ),
      )
  ]
  assert [
      i
      for i in fresh.requests[0]['input']
      if i.get('id') in {x['id'] for x in raw}
  ] == raw
  assert any(
      i.get('phase') == 'final_answer' for i in fresh.requests[0]['input']
  )
  await sessions.create_session(
      app_name='native', user_id='other', session_id='separate'
  )
  _ = [
      e
      async for e in runner.run_async(
          user_id='other',
          session_id='separate',
          new_message=types.Content(
              role='user', parts=[types.Part(text='Hello.')]
          ),
      )
  ]
  assert not any(i['type'] == 'reasoning' for i in fresh.requests[1]['input'])
  await runner.close()
  await sessions.close()


@pytest.mark.asyncio
async def test_compaction_retains_recent_native_state_and_foreign_agent_fencing_discards_opaque_state():
  old = _response(
      response([reasoning('old'), message('old-message')], 'old'), MODEL, False
  ).content
  recent_items = [reasoning('recent-1'), reasoning('recent-2'), call()]
  recent = _response(response(recent_items, 'recent'), MODEL, False).content
  events = [
      Event(author='agent', timestamp=1, content=old),
      Event(
          author='user',
          timestamp=2,
          actions=EventActions(
              compaction=EventCompaction(
                  start_timestamp=1,
                  end_timestamp=1,
                  compacted_content=types.Content(
                      role='model',
                      parts=[types.Part(text='Summary of the old request.')],
                  ),
              )
          ),
      ),
      Event(author='agent', timestamp=3, content=recent),
      Event(
          author='agent',
          timestamp=4,
          content=types.Content(
              role='user',
              parts=[
                  types.Part(
                      function_response=types.FunctionResponse(
                          id='call_lookup',
                          name='lookup',
                          response={'value': 42},
                      )
                  )
              ],
          ),
      ),
  ]
  events = [Event.model_validate_json(e.model_dump_json()) for e in events]
  contents = _get_contents(
      None, events, 'agent', preserve_function_call_ids=True
  )
  client = Client([response([message()])])
  model = LiteLlm(model=MODEL, use_responses_api=True, llm_client=client)
  _ = [r async for r in model.generate_content_async(request(contents))]
  native = [
      i
      for i in client.requests[0]['input']
      if i['type'] in ('reasoning', 'function_call')
  ]
  assert native == recent_items
  assert any(
      i['type'] == 'function_call_output' and i['call_id'] == 'call_lookup'
      for i in client.requests[0]['input']
  )
  foreign = _get_contents(
      None, [events[2]], 'coordinator', preserve_function_call_ids=True
  )
  assert foreign[0].role == 'user'
  assert all(
      NATIVE_RESPONSE_KEY not in (p.part_metadata or {})
      and 'opaque-' not in (p.text or '')
      for c in foreign
      for p in c.parts or []
  )
  reasoning_only = _response(
      response([reasoning('only')], 'only'), MODEL, False
  ).content
  assert (
      not reasoning_only.parts[0].text
      and not reasoning_only.parts[0].thought_signature
  )
  kept = _get_contents(
      None,
      [Event(author='agent', content=reasoning_only)],
      'agent',
      preserve_function_call_ids=True,
  )
  assert kept[0].parts[0].part_metadata[NATIVE_RESPONSE_KEY]['output'] == [
      reasoning('only')
  ]


@pytest.mark.asyncio
async def test_confirmation_pause_database_restart_and_resume_preserve_reasoning_and_call_ids(
    tmp_path,
):
  effects = []

  def change(value: int) -> dict:
    """Apply a synthetic effect after native confirmation."""
    effects.append(value)
    return {'value': value}

  tool = FunctionTool(change, require_confirmation=True)
  original = [
      reasoning('approved-1'),
      reasoning('approved-2'),
      call('change', 'call_change', '{"value":42}'),
  ]
  client = Client([response(original)])
  root = LlmAgent(
      name='agent',
      model=LiteLlm(model=MODEL, use_responses_api=True, llm_client=client),
      tools=[tool],
  )
  url = f'sqlite+aiosqlite:///{tmp_path / "pause.db"}'
  sessions = DatabaseSessionService(db_url=url)
  await sessions.create_session(
      app_name='native', user_id='user', session_id='chat'
  )
  runner = Runner(
      app=App(
          name='native',
          root_agent=root,
          resumability_config=ResumabilityConfig(is_resumable=True),
      ),
      session_service=sessions,
  )
  paused = [
      e
      async for e in runner.run_async(
          user_id='user',
          session_id='chat',
          new_message=types.Content(
              role='user', parts=[types.Part(text='Change it.')]
          ),
      )
  ]
  assert effects == [] and len(client.requests) == 1
  pending = next(
      (e, fc)
      for e in paused
      for fc in e.get_function_calls()
      if fc.name == 'adk_request_confirmation'
  )
  await runner.close()
  await sessions.close()
  fresh = Client([response([message()], 'resumed')])
  restarted = root.clone(
      update={
          'model': LiteLlm(
              model=MODEL, use_responses_api=True, llm_client=fresh
          )
      }
  )
  sessions = DatabaseSessionService(db_url=url)
  runner = Runner(
      app=App(
          name='native',
          root_agent=restarted,
          resumability_config=ResumabilityConfig(is_resumable=True),
      ),
      session_service=sessions,
  )
  answer = types.Content(
      role='user',
      parts=[
          types.Part(
              function_response=types.FunctionResponse(
                  name='adk_request_confirmation',
                  id=pending[1].id,
                  response={'confirmed': True},
              )
          )
      ],
  )
  _ = [
      e
      async for e in runner.run_async(
          user_id='user',
          session_id='chat',
          invocation_id=pending[0].invocation_id,
          new_message=answer,
      )
  ]
  assert effects == [42]
  assert [
      i
      for i in fresh.requests[0]['input']
      if i.get('id') in {x['id'] for x in original}
  ] == original
  assert any(
      i['type'] == 'function_call_output' and i['call_id'] == 'call_change'
      for i in fresh.requests[0]['input']
  )
  await runner.close()
  await sessions.close()


@pytest.mark.asyncio
async def test_native_runner_compaction_persists_summary_outputs_and_retains_recent_reasoning(
    tmp_path,
):
  old = _response(
      response(
          [
              reasoning('old'),
              message('old-answer', 'The old case has 42 units.'),
          ],
          'old',
      ),
      MODEL,
      False,
  ).content
  recent_items = [reasoning('recent-1'), reasoning('recent-2'), call()]
  recent = _response(response(recent_items, 'recent'), MODEL, False).content
  summary_items = [
      reasoning('summary-reasoning'),
      message('summary-message', 'The old case has 42 units.'),
  ]
  summary_client = Client([response(summary_items, 'summary-response')])
  summary_model = LiteLlm(
      model=MODEL, use_responses_api=True, llm_client=summary_client
  )
  client = Client([response([message('answer')], 'new-answer')])
  root = LlmAgent(
      name='agent',
      model=LiteLlm(model=MODEL, use_responses_api=True, llm_client=client),
      tools=[lookup],
  )
  sessions = DatabaseSessionService(
      db_url=f'sqlite+aiosqlite:///{tmp_path / "compact.db"}'
  )
  session = await sessions.create_session(
      app_name='native', user_id='user', session_id='chat'
  )
  events = [
      Event(
          author='user',
          invocation_id='old',
          timestamp=1,
          content=types.Content(
              role='user', parts=[types.Part(text='Large old evidence ' * 1000)]
          ),
      ),
      Event(author='agent', invocation_id='old', timestamp=2, content=old),
      Event(
          author='agent', invocation_id='recent', timestamp=3, content=recent
      ),
      Event(
          author='agent',
          invocation_id='recent',
          timestamp=4,
          content=types.Content(
              role='user',
              parts=[
                  types.Part(
                      function_response=types.FunctionResponse(
                          id='call_lookup',
                          name='lookup',
                          response={'value': 42},
                      )
                  )
              ],
          ),
      ),
  ]
  for event in events:
    await sessions.append_event(session=session, event=event)
  config = EventsCompactionConfig(
      compaction_interval=10000,
      overlap_size=0,
      token_threshold=2000,
      event_retention_size=2,
      summarizer=LlmEventSummarizer(llm=summary_model),
  )
  runner = Runner(
      app=App(name='native', root_agent=root, events_compaction_config=config),
      session_service=sessions,
  )
  _ = [
      e
      async for e in runner.run_async(
          user_id='user',
          session_id='chat',
          new_message=types.Content(
              role='user',
              parts=[types.Part(text='Continue from the recent lookup.')],
          ),
      )
  ]
  assert len(summary_client.requests) == 1
  assert 'opaque-' not in json.dumps(summary_client.requests[0]['input'])
  items = client.requests[0]['input']
  assert [i for i in items if i['type'] == 'reasoning'] == [
      summary_items[0],
      recent_items[0],
      recent_items[1],
  ]
  assert any(
      i['type'] == 'function_call_output' and i['call_id'] == 'call_lookup'
      for i in items
  )
  stored = await sessions.get_session(
      app_name='native', user_id='user', session_id='chat'
  )
  summary_event = next(e for e in stored.events if e.actions.compaction)
  metadata = summary_event.actions.compaction.compacted_content.parts[
      0
  ].part_metadata[NATIVE_RESPONSE_KEY]
  assert metadata['output'] == summary_items
  await runner.close()
  await sessions.close()
