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

from typing import Annotated
from typing import Literal

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field


class CustomToolTextFormat(BaseModel):
  """Unconstrained raw text input for a custom tool."""

  model_config = ConfigDict(extra='forbid')
  type: Literal['text'] = 'text'


class CustomToolGrammarFormat(BaseModel):
  """A native context-free grammar applied by the model provider."""

  model_config = ConfigDict(extra='forbid')
  type: Literal['grammar'] = 'grammar'
  syntax: Literal['lark', 'regex']
  definition: str = Field(min_length=1)


CustomToolInputFormat = Annotated[
    CustomToolTextFormat | CustomToolGrammarFormat, Field(discriminator='type')
]


class CustomToolDeclaration(BaseModel):
  """A native freeform tool declaration, separate from function JSON schemas."""

  model_config = ConfigDict(extra='forbid')
  type: Literal['custom'] = 'custom'
  name: str = Field(min_length=1)
  description: str
  format: CustomToolInputFormat = Field(default_factory=CustomToolTextFormat)
