# utils/llm_clients/citation_stream_buffer.py
"""
GPT-Style Citation Stream Buffer.

Sits between the LLM token stream and the WebSocket broadcaster inside
StreamingChatManager._stream_rag_response(). Intercepts citation patterns
like [Doc.pdf, p. 5] and replaces them with a {{CITE}} placeholder in the
broadcast stream, so the user never sees raw bracket markup during streaming.

The actual citation data still flows through the existing
_extract_citations_from_accumulated_text → _broadcast_citations pipeline.

States:
    PASSTHROUGH  — characters flow straight to output
    BUFFERING    — saw '[', holding chars until pattern completes or fails

On match:  yields "{{CITE}}" placeholder
On bail:   yields the buffered characters as plain text
"""

import re
import logging

logger = logging.getLogger(__name__)


class CitationStreamBuffer:

    MAX_BUFFER = 200  # bail-out length — real citations are <150 chars

    # Matches the same patterns as citation_processor.extract_inline_citations_from_content():
    #   [Doc.pdf, p. 5]       single page
    #   [Doc.pdf, pp. 3-4]    multi page range
    #   [Doc.pdf, Page 5]     alternate keyword
    _CITATION_RE = re.compile(
        r'^\[([^,\]]+),\s*(?:pp?\.|Page)\s*([\d,\-\s]+)\]$'
    )

    def __init__(self):
        self._state = "PASSTHROUGH"
        self._held = ""

    def feed(self, text: str):
        """
        Feed a chunk of text from the LLM. Yields zero or more strings
        that are safe to broadcast:
          - plain text segments
          - "{{CITE}}" placeholders where citations were suppressed
        """
        if not text:
            return

        for char in text:
            if self._state == "PASSTHROUGH":
                if char == "[":
                    self._state = "BUFFERING"
                    self._held = char
                else:
                    yield char

            elif self._state == "BUFFERING":
                self._held += char

                if char == "]":
                    # Closing bracket — check if held text is a citation
                    if self._CITATION_RE.match(self._held):
                        logger.debug(f"🏷️ Citation buffered & replaced: {self._held}")
                        yield "{{CITE}}"
                    else:
                        # Not a citation pattern — flush as plain text
                        logger.debug(f"📝 Bracket was not a citation: {self._held}")
                        yield self._held
                    self._held = ""
                    self._state = "PASSTHROUGH"

                elif len(self._held) > self.MAX_BUFFER:
                    # Too long to be a citation — bail out
                    logger.debug(f"📝 Buffer overflow, flushing: {self._held[:50]}...")
                    yield self._held
                    self._held = ""
                    self._state = "PASSTHROUGH"

                elif char == "[" and len(self._held) > 1:
                    # Nested bracket — first '[' wasn't a citation start
                    # Flush everything before this new '[', restart buffering
                    yield self._held[:-1]
                    self._held = char  # keep the new '['

    def finalize(self):
        """
        Call at end of stream. Flushes any remaining buffered content
        as plain text (handles LLM stopping mid-citation).
        """
        if self._held:
            logger.debug(f"📝 Finalizing buffer with leftover: {self._held}")
            yield self._held
            self._held = ""
        self._state = "PASSTHROUGH"

    @property
    def is_buffering(self) -> bool:
        return self._state == "BUFFERING"
