"""Tool calling tests for Response API.

Tests for function calling functionality and tool choices across different backends.

Source: Migrated from e2e_response_api/features/test_tools_call.py
"""

from __future__ import annotations

import json
import logging
import time

import pytest

logger = logging.getLogger(__name__)


# =============================================================================
# Shared Tool Definitions
# =============================================================================


SYSTEM_DIAGNOSTICS_FUNCTION = {
    "type": "function",
    "name": "get_system_diagnostics",
    "description": "Retrieve real-time diagnostics for a spacecraft system.",
    "parameters": {
        "type": "object",
        "properties": {
            "system_name": {
                "type": "string",
                "description": "Name of the spacecraft system to query. "
                "Example: 'Astra-7 Core Reactor'.",
            }
        },
        "required": ["system_name"],
    },
}

GET_WEATHER_FUNCTION = {
    "type": "function",
    "name": "get_weather",
    "description": "Get the current weather in a given location",
    "parameters": {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "The city name, e.g., San Francisco",
            }
        },
        "required": ["location"],
    },
}

CALCULATE_FUNCTION = {
    "type": "function",
    "name": "calculate",
    "description": "Perform a mathematical calculation",
    "parameters": {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "The mathematical expression to evaluate",
            }
        },
        "required": ["expression"],
    },
}

SEARCH_WEB_FUNCTION = {
    "type": "function",
    "name": "search_web",
    "description": "Search the web for information",
    "parameters": {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    },
}

LOCAL_SEARCH_FUNCTION = {
    "type": "function",
    "name": "local_search",
    "description": "Search local database",
    "parameters": {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    },
}

GET_HOROSCOPE_FUNCTION = {
    "type": "function",
    "name": "get_horoscope",
    "description": "Get today's horoscope for an astrological sign.",
    "parameters": {
        "type": "object",
        "properties": {
            "sign": {
                "type": "string",
                "description": "An astrological sign like Taurus or Aquarius",
            },
        },
        "required": ["sign"],
    },
}



# =============================================================================
# Cloud Backend Tests (OpenAI) - Basic Function Calling
# =============================================================================


@pytest.mark.parametrize("setup_backend", ["openai"], indirect=True)
class TestToolCallingCloud:
    """Tool calling tests against cloud APIs."""

    def test_basic_function_call(self, setup_backend):
        """Test basic function calling workflow."""
        _, model, client, gateway = setup_backend

        tools = [GET_HOROSCOPE_FUNCTION]
        system_prompt = (
            "You are a helpful assistant that can call functions. "
            "When a user asks for horoscope information, call the function. "
            "IMPORTANT: Don't reply directly to the user, only call the function. "
        )

        input_list = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "What is my horoscope? I am an Aquarius."},
        ]

        resp = client.responses.create(model=model, input=input_list, tools=tools)

        assert resp.error is None
        assert resp.id is not None
        assert resp.status == "completed"
        assert resp.output is not None

        output = resp.output
        assert isinstance(output, list)
        assert len(output) > 0

        # Check for function_call in output
        function_calls = [item for item in output if item.type == "function_call"]
        assert (
            len(function_calls) > 0
        ), "Response should contain at least one function_call"

        # Verify function_call structure
        function_call = function_calls[0]
        assert function_call.call_id is not None
        assert function_call.name == "get_horoscope"
        assert function_call.arguments is not None

        # Parse arguments
        args = json.loads(function_call.arguments)
        assert "sign" in args
        assert args["sign"].lower() == "aquarius"

        # Provide function call output
        input_list.append(function_call)
        horoscope = f"{args['sign']}: Next Tuesday you will befriend a baby otter."
        input_list.append(
            {
                "type": "function_call_output",
                "call_id": function_call.call_id,
                "output": json.dumps({"horoscope": horoscope}),
            }
        )

        # Second request with function output
        resp2 = client.responses.create(
            model=model,
            input=input_list,
            instructions="Respond only with a horoscope generated by a tool.",
            tools=tools,
        )
        assert resp2.error is None
        assert resp2.status == "completed"

        output2 = resp2.output
        assert len(output2) > 0

        messages = [item for item in output2 if item.type == "message"]
        assert len(messages) > 0

        message = messages[0]
        assert message.content is not None
        text_parts = [
            part.text for part in message.content if part.type == "output_text"
        ]
        full_text = " ".join(text_parts).lower()
        assert "baby otter" in full_text or "aquarius" in full_text

# =============================================================================
# Local Backend Tests (gRPC with Harmony model) - Tool Choice
# =============================================================================


@pytest.mark.e2e
@pytest.mark.model("gpt-oss")
@pytest.mark.gateway(
    extra_args=["--reasoning-parser=gpt-oss"]
)
@pytest.mark.parametrize("setup_backend", ["grpc"], indirect=True)
class TestToolChoiceHarmony:
    """Tool choice tests against local gRPC backend with Harmony model."""

    def test_tool_choice_auto(self, setup_backend):
        """Test tool_choice="auto" allows model to decide whether to use tools."""
        _, model, client, gateway = setup_backend

        tools = [GET_WEATHER_FUNCTION]

        resp = client.responses.create(
            model=model,
            input="What is the weather in Seattle?",
            tools=tools,
            tool_choice="auto",
            stream=False,
        )

        assert resp.id is not None
        assert resp.error is None

        output = resp.output
        assert len(output) > 0

        function_calls = [item for item in output if item.type == "function_call"]
        assert (
            len(function_calls) > 0
        ), "Model should choose to call function with tool_choice='auto'"

    def test_tool_choice_required(self, setup_backend):
        """Test tool_choice="required" forces the model to call at least one tool."""
        _, model, client, gateway = setup_backend

        tools = [CALCULATE_FUNCTION]

        resp = client.responses.create(
            model=model,
            input="What is 15 * 23?",
            tools=tools,
            tool_choice="required",
            stream=False,
        )

        assert resp.id is not None
        assert resp.error is None

        output = resp.output
        function_calls = [item for item in output if item.type == "function_call"]
        assert (
            len(function_calls) > 0
        ), "tool_choice='required' must force at least one function call"

    def test_tool_choice_specific_function(self, setup_backend):
        """Test tool_choice with specific function name forces that function to be called."""
        _, model, client, gateway = setup_backend

        tools = [SEARCH_WEB_FUNCTION, GET_WEATHER_FUNCTION]

        resp = client.responses.create(
            model=model,
            input="What's happening in the news today?",
            tools=tools,
            tool_choice={"type": "function", "function": {"name": "search_web"}},
            stream=False,
        )

        assert resp.id is not None
        assert resp.error is None

        output = resp.output
        function_calls = [item for item in output if item.type == "function_call"]
        assert len(function_calls) > 0, "Must call the specified function"
        assert (
            function_calls[0].name == "search_web"
        ), "Must call the function specified in tool_choice"

    def test_tool_choice_streaming(self, setup_backend):
        """Test tool_choice parameter works correctly with streaming."""
        _, model, client, gateway = setup_backend

        tools = [CALCULATE_FUNCTION]

        resp = client.responses.create(
            model=model,
            input="Calculate 42 * 17",
            tools=tools,
            tool_choice="required",
            stream=True,
        )

        events = list(resp)
        assert len(events) > 0

        event_types = [e.type for e in events]
        assert "response.function_call_arguments.delta" in event_types

        completed_events = [e for e in events if e.type == "response.completed"]
        assert len(completed_events) == 1

        output = completed_events[0].response.output
        function_calls = [item for item in output if item.type == "function_call"]
        assert len(function_calls) > 0

    def test_basic_function_call(self, setup_backend):
        """Test basic function calling workflow."""
        _, model, client, gateway = setup_backend

        tools = [GET_HOROSCOPE_FUNCTION]
        system_prompt = (
            "You are a helpful assistant that can call functions. "
            "When a user asks for horoscope information, call the function. "
            "IMPORTANT: Don't reply directly to the user, only call the function. "
        )

        input_list = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "What is my horoscope? I am an Aquarius."},
        ]

        resp = client.responses.create(model=model, input=input_list, tools=tools)

        assert resp.error is None
        assert resp.id is not None
        assert resp.status == "completed"

        output = resp.output
        function_calls = [item for item in output if item.type == "function_call"]
        assert len(function_calls) > 0

        function_call = function_calls[0]
        assert function_call.name == "get_horoscope"

        args = json.loads(function_call.arguments)
        assert "sign" in args
        assert args["sign"].lower() == "aquarius"

# =============================================================================
# Local Backend Tests (gRPC with Qwen model) - Tool Choice
# =============================================================================


@pytest.mark.e2e
@pytest.mark.model("qwen-14b")
@pytest.mark.gateway(
    extra_args=["--tool-call-parser", "qwen"]
)
@pytest.mark.parametrize("setup_backend", ["grpc"], indirect=True)
class TestToolChoiceLocal:
    """Tool choice tests against local gRPC backend with Qwen model."""

    def test_tool_choice_auto(self, setup_backend):
        """Test tool_choice="auto" allows model to decide whether to use tools."""
        _, model, client, gateway = setup_backend

        tools = [GET_WEATHER_FUNCTION]

        resp = client.responses.create(
            model=model,
            input="What is the weather in Seattle?",
            tools=tools,
            tool_choice="auto",
            stream=False,
        )

        assert resp.id is not None
        assert resp.error is None

        output = resp.output
        assert len(output) > 0

        function_calls = [item for item in output if item.type == "function_call"]
        assert len(function_calls) > 0

    def test_tool_choice_required(self, setup_backend):
        """Test tool_choice="required" forces the model to call at least one tool."""
        _, model, client, gateway = setup_backend

        tools = [CALCULATE_FUNCTION]

        resp = client.responses.create(
            model=model,
            input="What is 15 * 23?",
            tools=tools,
            tool_choice="required",
            stream=False,
        )

        assert resp.id is not None
        assert resp.error is None

        function_calls = [item for item in resp.output if item.type == "function_call"]
        assert len(function_calls) > 0

    def test_tool_choice_specific_function(self, setup_backend):
        """Test tool_choice with specific function name forces that function to be called."""
        _, model, client, gateway = setup_backend

        tools = [SEARCH_WEB_FUNCTION, GET_WEATHER_FUNCTION]

        resp = client.responses.create(
            model=model,
            input="What's happening in the news today?",
            tools=tools,
            tool_choice={"type": "function", "function": {"name": "search_web"}},
            stream=False,
        )

        assert resp.id is not None
        assert resp.error is None

        function_calls = [item for item in resp.output if item.type == "function_call"]
        assert len(function_calls) > 0
        assert function_calls[0].name == "search_web"
