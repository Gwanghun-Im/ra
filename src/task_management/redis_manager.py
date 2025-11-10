"""Redis Task Manager - High-level interface for centralized task management"""

import logging
from typing import Optional, List, Dict, Any
from datetime import datetime

from a2a.types import Task, TaskState, TaskStatus, Message
from a2a.server.tasks import TaskManager
from .redis_task_store import RedisTaskStore
from .models import TaskType, TaskPriority

logger = logging.getLogger(__name__)


class RedisTaskManager(TaskManager):
    """
    High-level manager for centralized task tracking using Redis.

    This class provides a convenient interface for:
    - Creating and tracking tasks across all agents
    - Managing task queues by priority
    - Querying task history by user, context, or agent
    - Real-time task event notifications
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        ttl_days: int = 30,
    ):
        """
        Initialize Redis Task Manager

        Args:
            redis_url: Redis connection URL
            ttl_days: Days to keep completed tasks
        """
        self.task_store = RedisTaskStore(redis_url=redis_url, ttl_days=ttl_days)
        self._initialized = False
        logger.info("RedisTaskManager initialized")

    async def initialize(self) -> None:
        """Initialize the task manager"""
        if not self._initialized:
            await self.task_store.initialize()
            self._initialized = True
            logger.info("RedisTaskManager ready")

    async def close(self) -> None:
        """Close the task manager"""
        await self.task_store.close()
        self._initialized = False

    async def create_task(
        self,
        task_id: str,
        message: Message,
        user_id: str = "default",
        context_id: Optional[str] = None,
        task_type: TaskType = TaskType.GENERAL,
        priority: TaskPriority = TaskPriority.NORMAL,
        assigned_agent: Optional[str] = None,
    ) -> Task:
        """
        Create a new task and save to Redis

        Args:
            task_id: Unique task identifier
            message: A2A Message object
            user_id: User identifier
            context_id: Conversation context ID
            task_type: Type of task (for classification)
            priority: Task priority
            assigned_agent: Agent to assign the task to

        Returns:
            Created Task object
        """
        if not self._initialized:
            await self.initialize()

        # Create A2A Task object
        task = Task(
            id=task_id,
            message=message,
            context_id=context_id,
            status=TaskStatus(
                state=TaskState.submitted,
                message="Task created",
            ),
        )

        # Save task to Redis
        await self.task_store.save(task)

        # Save extended metadata
        await self.task_store.save_metadata(
            task_id=task_id,
            user_id=user_id,
            context_id=context_id,
            task_type=task_type,
            priority=priority,
            assigned_agent=assigned_agent,
        )

        # Enqueue task for processing
        await self.task_store.enqueue_task(task_id, priority)

        # Publish task creation event
        await self.task_store.publish_task_event(
            task_id=task_id,
            event_type="task_created",
            data={
                "user_id": user_id,
                "task_type": task_type.value,
                "priority": priority.value,
                "assigned_agent": assigned_agent,
            },
        )

        logger.info(
            f"Task {task_id} created for user {user_id}, "
            f"type={task_type.value}, priority={priority.value}"
        )

        return task

    async def update_task(
        self,
        task_id: str,
        state: TaskState,
        message: Optional[str] = None,
    ) -> None:
        """
        Update task status

        Args:
            task_id: Task ID to update
            state: New task state
            message: Optional status message
        """
        if not self._initialized:
            await self.initialize()

        # Get existing task
        task = await self.task_store.get(task_id)
        if not task:
            logger.warning(f"Task {task_id} not found for update")
            return

        # Update status
        task.status = TaskStatus(
            state=state,
            message=message or task.status.message,
        )

        # Save updated task
        await self.task_store.save(task)

        # Publish status update event
        await self.task_store.publish_task_event(
            task_id=task_id,
            event_type="task_status_updated",
            data={
                "state": state.name,
                "message": message,
            },
        )

        logger.info(f"Task {task_id} updated to state {state.name}")

    async def get_task(self, task_id: str) -> Optional[Task]:
        """Get a task by ID"""
        if not self._initialized:
            await self.initialize()

        return await self.task_store.get(task_id)

    async def get_task_with_metadata(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Get task along with extended metadata

        Returns:
            Dictionary with 'task' and 'metadata' keys
        """
        if not self._initialized:
            await self.initialize()

        task = await self.task_store.get(task_id)
        if not task:
            return None

        metadata = await self.task_store.get_metadata(task_id)

        return {
            "task": task,
            "metadata": metadata or {},
        }

    async def get_user_tasks(self, user_id: str, limit: int = 50, offset: int = 0) -> List[Task]:
        """Get all tasks for a user"""
        if not self._initialized:
            await self.initialize()

        return await self.task_store.get_user_tasks(user_id, limit, offset)

    async def get_context_tasks(self, context_id: str, limit: int = 50) -> List[Task]:
        """Get all tasks for a conversation context"""
        if not self._initialized:
            await self.initialize()

        return await self.task_store.get_context_tasks(context_id, limit)

    async def get_agent_tasks(self, agent_name: str, limit: int = 50) -> List[Task]:
        """Get all tasks assigned to an agent"""
        if not self._initialized:
            await self.initialize()

        return await self.task_store.get_agent_tasks(agent_name, limit)

    async def get_user_task_summary(self, user_id: str) -> Dict[str, Any]:
        """
        Get summary statistics for a user's tasks

        Returns:
            Dictionary with task counts by state and type
        """
        if not self._initialized:
            await self.initialize()

        tasks = await self.task_store.get_user_tasks(user_id, limit=1000)

        summary = {
            "total": len(tasks),
            "by_state": {},
            "by_type": {},
            "recent_tasks": [],
        }

        for task in tasks[:10]:  # Recent 10 tasks
            metadata = await self.task_store.get_metadata(task.id)

            # Count by state
            state_name = task.status.state.name
            summary["by_state"][state_name] = summary["by_state"].get(state_name, 0) + 1

            # Count by type
            if metadata:
                task_type = metadata.get("task_type", "unknown")
                summary["by_type"][task_type] = summary["by_type"].get(task_type, 0) + 1

            # Add to recent tasks
            summary["recent_tasks"].append(
                {
                    "task_id": task.id,
                    "state": state_name,
                    "type": metadata.get("task_type") if metadata else None,
                    "created_at": metadata.get("created_at") if metadata else None,
                }
            )

        return summary

    async def delete_task(self, task_id: str) -> None:
        """Delete a task"""
        if not self._initialized:
            await self.initialize()

        await self.task_store.delete(task_id)
        logger.info(f"Task {task_id} deleted")

    async def get_next_task(self, priority: TaskPriority = TaskPriority.NORMAL) -> Optional[str]:
        """
        Get next task ID from priority queue

        Args:
            priority: Priority level to dequeue from

        Returns:
            Task ID if available, None otherwise
        """
        if not self._initialized:
            await self.initialize()

        return await self.task_store.dequeue_task(priority)

    async def get_queue_size(self, priority: TaskPriority = TaskPriority.NORMAL) -> int:
        """Get number of tasks in queue for given priority"""
        if not self._initialized:
            await self.initialize()

        return await self.task_store.get_queue_size(priority)

    async def get_all_queue_sizes(self) -> Dict[str, int]:
        """Get queue sizes for all priorities"""
        if not self._initialized:
            await self.initialize()

        sizes = {}
        for priority in TaskPriority:
            size = await self.task_store.get_queue_size(priority)
            sizes[priority.name] = size

        return sizes

    @classmethod
    async def create(
        cls,
        redis_url: str = "redis://localhost:6379",
        ttl_days: int = 30,
    ) -> "RedisTaskManager":
        """
        Factory method to create and initialize RedisTaskManager

        Args:
            redis_url: Redis connection URL
            ttl_days: Days to keep completed tasks

        Returns:
            Initialized RedisTaskManager instance
        """
        manager = cls(redis_url=redis_url, ttl_days=ttl_days)
        await manager.initialize()
        return manager
