"""
ChatbotService — orchestrates the LangGraph workflow and maps its output
to the API response schema.
"""

from datetime import datetime
from typing import Any, Dict

from app.chatbot_workflow import compiled_graph
from app.core.logging_config import get_logger
from app.models.schemas import ChatResponse

logger = get_logger("aura.service.chatbot")


class ChatbotService:
    def __init__(self) -> None:
        self._graph = compiled_graph

    def process_message(
        self,
        message: str,
        user_id: str,
        conversation_state: Dict[str, Any],
    ) -> ChatResponse:
        """
        Run the message through the AURA LangGraph state machine.

        The graph executes three sequential nodes:
          intent_analysis  → data_retrieval → appointment_trigger

        Returns a ChatResponse with the natural-language reply, detected intent,
        confidence score, and updated conversation state.
        """
        initial_state = {
            "query": message,
            "conversation_state": {**conversation_state, "user_id": user_id},
            "intent": "",
            "confidence": 0.0,
            "response": "",
            "appointment_action": "",
            "datetime": "",
        }

        try:
            result = self._graph.invoke(initial_state)
        except Exception as exc:
            logger.error(
                "LangGraph workflow failed",
                extra={"user_id": user_id, "error": str(exc)},
                exc_info=True,
            )
            return ChatResponse(
                response=(
                    "I encountered a technical issue processing your request. "
                    "Please try again or rephrase your message."
                ),
                intent="error",
                confidence=0.0,
                conversation_state=conversation_state,
                timestamp=datetime.now(),
            )

        response_text = result.get(
            "response", "I'm sorry, I didn't quite understand that. Could you rephrase?"
        )
        intent = result.get("intent", "unknown")
        confidence = result.get("confidence", 0.5)
        updated_state = result.get("conversation_state", conversation_state)

        logger.info(
            "Workflow completed",
            extra={
                "user_id": user_id,
                "intent": intent,
                "confidence": round(float(confidence), 3),
            },
        )

        return ChatResponse(
            response=response_text,
            intent=intent,
            confidence=float(confidence),
            conversation_state=updated_state,
            timestamp=datetime.now(),
        )
