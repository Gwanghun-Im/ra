"""A2A 메시지 처리를 위한 유틸리티 함수들"""

import uuid
from typing import List
from a2a.types import (
    Message,
    TextPart,
    Part,
    Artifact,
    TaskStatusUpdateEvent,
    TaskStatus,
)
from a2a.server.events.event_queue import EventQueue


def extract_text_from_message(message: Message) -> str:
    """
    메시지의 모든 part에서 텍스트를 추출

    Args:
        message: A2A Message 객체

    Returns:
        추출된 텍스트 (여러 part의 텍스트를 공백으로 연결)
    """
    content_parts = []
    for part in message.parts:
        # Part는 Union 타입이므로 root를 통해 실제 객체에 접근
        actual_part = part.root if hasattr(part, 'root') else part
        if hasattr(actual_part, 'text'):
            content_parts.append(actual_part.text)
    return " ".join(content_parts)


def create_text_response_message(
    response_text: str,
    context_id: str | None = None,
    message_id: str | None = None,
) -> Message:
    """
    텍스트 응답을 위한 Message 객체 생성

    Args:
        response_text: 응답 텍스트
        context_id: 컨텍스트 ID (없으면 새로 생성)
        message_id: 메시지 ID (없으면 새로 생성)

    Returns:
        생성된 Message 객체
    """
    return Message(
        role="agent",
        parts=[TextPart(text=response_text)],
        message_id=message_id or str(uuid.uuid4()),
        context_id=context_id,
    )


def create_response_from_request(
    request_message: Message,
    response_text: str,
) -> Message:
    """
    요청 메시지로부터 응답 메시지 생성 (context_id 유지)

    Args:
        request_message: 원본 요청 메시지
        response_text: 응답 텍스트

    Returns:
        생성된 응답 Message 객체
    """
    return create_text_response_message(
        response_text=response_text,
        context_id=request_message.context_id,
    )


async def enqueue_response_as_artifact(
    event_queue: EventQueue,
    task_id: str,
    context_id: str,
    response_text: str,
    final: bool = True,
) -> None:
    """
    응답 텍스트를 Artifact로 감싸서 event queue에 추가

    Args:
        event_queue: A2A EventQueue 객체
        task_id: Task ID
        context_id: Context ID
        response_text: 응답 텍스트
        final: 최종 응답 여부
    """
    # Create artifact with response text
    artifact = Artifact(
        artifact_id=str(uuid.uuid4()),
        parts=[TextPart(text=response_text)],
    )

    # Create artifact update event
    from a2a.types import TaskArtifactUpdateEvent

    artifact_event = TaskArtifactUpdateEvent(
        task_id=task_id,
        context_id=context_id,
        artifact=artifact,
        last_chunk=final,
    )

    # Enqueue the artifact event
    await event_queue.enqueue_event(artifact_event)

    # If this is the final response, also send status update
    if final:
        from a2a.types import TaskState

        status_event = TaskStatusUpdateEvent(
            task_id=task_id,
            context_id=context_id,
            status=TaskStatus(state=TaskState.completed),
            final=True,
        )
        await event_queue.enqueue_event(status_event)
