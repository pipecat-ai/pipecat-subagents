#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import unittest

from pipecat.frames.frames import LLMContextFrame, TextFrame, TranscriptionFrame
from pipecat.processors.aggregators.llm_context import (
    LLMContext,
    LLMSpecificMessage,
    NotGiven,
)
from pipecat.processors.frame_processor import FrameDirection

from pipecat_subagents.bus.adapters import FrameAdapter
from pipecat_subagents.bus.messages import BusFrameMessage
from pipecat_subagents.bus.serializers import JSONMessageSerializer


class TestFrameAdapterTextFrame(unittest.TestCase):
    def test_round_trip(self):
        adapter = FrameAdapter()
        frame = TextFrame(text="hello world")
        data = adapter.serialize(frame)
        restored = adapter.deserialize(data)

        self.assertIsInstance(restored, TextFrame)
        self.assertEqual(restored.text, "hello world")


class TestFrameAdapterTranscriptionFrame(unittest.TestCase):
    def test_round_trip_basic(self):
        adapter = FrameAdapter()
        frame = TranscriptionFrame(
            text="hello",
            user_id="user-1",
            timestamp="2026-03-17T00:00:00Z",
        )
        data = adapter.serialize(frame)
        restored = adapter.deserialize(data)

        self.assertIsInstance(restored, TranscriptionFrame)
        self.assertEqual(restored.text, "hello")
        self.assertEqual(restored.user_id, "user-1")
        self.assertEqual(restored.timestamp, "2026-03-17T00:00:00Z")
        self.assertIsNone(restored.language)
        self.assertFalse(restored.finalized)

    def test_round_trip_with_language(self):
        from pipecat.transcriptions.language import Language

        adapter = FrameAdapter()
        frame = TranscriptionFrame(
            text="hola",
            user_id="user-1",
            timestamp="2026-03-17T00:00:00Z",
            language=Language.ES,
            finalized=True,
        )
        data = adapter.serialize(frame)
        restored = adapter.deserialize(data)

        self.assertEqual(restored.language, Language.ES)
        self.assertTrue(restored.finalized)


class TestFrameAdapterLLMContextFrame(unittest.TestCase):
    def _round_trip_context(self, ctx):
        adapter = FrameAdapter()
        frame = LLMContextFrame(context=ctx)
        data = adapter.serialize(frame)
        restored = adapter.deserialize(data)
        self.assertIsInstance(restored, LLMContextFrame)
        return restored.context

    def test_round_trip_messages_only(self):
        ctx = LLMContext(messages=[
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "hello"},
        ])
        restored = self._round_trip_context(ctx)

        self.assertEqual(len(restored.messages), 2)
        self.assertEqual(restored.messages[0]["role"], "system")
        self.assertEqual(restored.messages[1]["content"], "hello")
        self.assertIsInstance(restored.tools, NotGiven)
        self.assertIsInstance(restored.tool_choice, NotGiven)

    def test_round_trip_with_specific_message(self):
        ctx = LLMContext(messages=[
            {"role": "user", "content": "hi"},
            LLMSpecificMessage(llm="anthropic", message={"custom": "data"}),
        ])
        restored = self._round_trip_context(ctx)

        self.assertEqual(len(restored.messages), 2)
        self.assertIsInstance(restored.messages[0], dict)
        self.assertIsInstance(restored.messages[1], LLMSpecificMessage)
        self.assertEqual(restored.messages[1].llm, "anthropic")
        self.assertEqual(restored.messages[1].message, {"custom": "data"})

    def test_round_trip_with_tools(self):
        from pipecat.adapters.schemas.function_schema import FunctionSchema
        from pipecat.adapters.schemas.tools_schema import ToolsSchema

        tools = ToolsSchema(standard_tools=[
            FunctionSchema(
                name="get_weather",
                description="Get the weather",
                properties={"location": {"type": "string"}},
                required=["location"],
            ),
        ])
        ctx = LLMContext(
            messages=[{"role": "user", "content": "weather?"}],
            tools=tools,
        )
        restored = self._round_trip_context(ctx)

        self.assertNotIsInstance(restored.tools, NotGiven)
        self.assertEqual(len(restored.tools.standard_tools), 1)
        self.assertEqual(restored.tools.standard_tools[0].name, "get_weather")
        self.assertEqual(restored.tools.standard_tools[0].required, ["location"])

    def test_round_trip_with_tool_choice(self):
        ctx = LLMContext(
            messages=[{"role": "user", "content": "hi"}],
            tool_choice="auto",
        )
        restored = self._round_trip_context(ctx)

        self.assertEqual(restored.tool_choice, "auto")


class TestFrameAdapterWithSerializer(unittest.TestCase):
    """Integration: FrameAdapter registered on JSONMessageSerializer."""

    def setUp(self):
        from pipecat.frames.frames import Frame

        self.serializer = JSONMessageSerializer()
        frame_adapter = FrameAdapter()
        self.serializer.register_adapter(Frame, frame_adapter)

    def test_text_frame_message_round_trip(self):
        msg = BusFrameMessage(
            source="a",
            frame=TextFrame(text="hello"),
            direction=FrameDirection.DOWNSTREAM,
        )
        data = self.serializer.serialize(msg)
        restored = self.serializer.deserialize(data)

        self.assertIsInstance(restored.frame, TextFrame)
        self.assertEqual(restored.frame.text, "hello")

    def test_llm_context_frame_message_round_trip(self):
        ctx = LLMContext(messages=[
            {"role": "system", "content": "Be helpful"},
            {"role": "user", "content": "hi"},
        ])
        msg = BusFrameMessage(
            source="voice_agent",
            frame=LLMContextFrame(context=ctx),
            direction=FrameDirection.DOWNSTREAM,
        )
        data = self.serializer.serialize(msg)
        restored = self.serializer.deserialize(data)

        self.assertIsInstance(restored.frame, LLMContextFrame)
        self.assertEqual(len(restored.frame.context.messages), 2)
        self.assertEqual(restored.frame.context.messages[0]["role"], "system")

    def test_transcription_frame_message_round_trip(self):
        msg = BusFrameMessage(
            source="main",
            frame=TranscriptionFrame(
                text="hello",
                user_id="u1",
                timestamp="2026-03-17T00:00:00Z",
            ),
            direction=FrameDirection.DOWNSTREAM,
        )
        data = self.serializer.serialize(msg)
        restored = self.serializer.deserialize(data)

        self.assertIsInstance(restored.frame, TranscriptionFrame)
        self.assertEqual(restored.frame.text, "hello")
        self.assertEqual(restored.frame.user_id, "u1")


if __name__ == "__main__":
    unittest.main()
