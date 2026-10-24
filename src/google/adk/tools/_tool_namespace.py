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

"""Native Responses tool namespace and its internal dispatch identity."""

from __future__ import annotations

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field


class ToolNamespace(BaseModel):
  """A searchable group of related functions in the Responses API."""

  model_config = ConfigDict(frozen=True)
  name: str = Field(pattern=r'^[A-Za-z0-9_-]{1,64}$')
  description: str = Field(min_length=1)


def dispatch_name(namespace: str | None, name: str) -> str:
  """Adapt the native namespace/name pair to ADK's string call identity."""
  return f'{namespace}.{name}' if namespace else name
