"""Redis-based implementation of A2A TaskStore"""

import asyncio
import logging
from datetime import datetime
from typing import Optional

import redis.asyncio as aioredis
from a2a.server.tasks import TaskStore
from a2a.types import Task, TaskState
from a2a.server.context import ServerCallContext

from .models import TaskPriority

logger = logging.getLogger(__name__)


class RedisTaskStore(TaskStore):
    """
    Redis-based implementation of A2A TaskStore with extended tracking capabilities.

    Features:
    - Persistent task storage using Redis
    - Task queueing by priority
    - Task history tracking per user
    - Real-time status updates via Pub/Sub
    - Metadata tracking (task_type, priority, agent assignment)
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        ttl_days: int = 30,
        key_prefix: str = "a2a:task",
    ):
        """
        Initialize Redis Task Store

        Args:
            redis_url: Redis connection URL
            ttl_days: Days to keep completed tasks (default: 30)
            key_prefix: Prefix for Redis keys (default: "a2a:task")
        """
        self.redis_url = redis_url
        self.ttl_days = ttl_days
        self.key_prefix = key_prefix
        self.redis_client: Optional[aioredis.Redis] = None
        self.lock = asyncio.Lock()
        self._initialized = False

        logger.info(f"RedisTaskStore initialized with URL: {redis_url}")

    async def initialize(self) -> None:
        """Initialize Redis connection"""
        if self._initialized:
            return

        try:
            self.redis_client = await aioredis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True,
            )
            # Test connection
            await self.redis_client.ping()
            self._initialized = True
            logger.info("Redis connection established successfully")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise

    async def close(self) -> None:
        """Close Redis connection"""
        if self.redis_client:
            await self.redis_client.close()
            self._initialized = False
            logger.info("Redis connection closed")

    def _task_key(self, task_id: str) -> str:
        """Generate Redis key for task data"""
        yyddmmhhmmss = datetime.utcnow().strftime("%y%m%d%H%M%S")
        return f"{self.key_prefix}:{yyddmmhhmmss}:{task_id}"

    def _metadata_key(self, task_id: str) -> str:
        """Generate Redis key for task metadata"""
        return f"{self.key_prefix}:meta:{task_id}"

    def _context_tasks_key(self, context_id: str) -> str:
        """Generate Redis key for context's task list"""
        return f"{self.key_prefix}:context:{context_id}:tasks"

    def _agent_tasks_key(self, agent_name: str) -> str:
        """Generate Redis key for agent's task list"""
        return f"{self.key_prefix}:agent:{agent_name}:tasks"

    async def save(self, task: Task, context: ServerCallContext | None = None) -> None:
        """
        Save or update a task in Redis

        Args:
            task: The A2A Task object to save
            context: Optional gRPC server call context
        """
        if not self._initialized:
            await self.initialize()

        async with self.lock:
            try:
                # Serialize task to JSON (A2A Task is a Pydantic model)
                task_data = task.model_dump_json()

                # Save task data
                task_key = self._task_key(task.id)
                await self.redis_client.set(task_key, task_data)

                # Set TTL for completed tasks
                if task.status.state in [
                    TaskState.completed,
                    TaskState.failed,
                    TaskState.canceled,
                ]:
                    ttl_seconds = self.ttl_days * 24 * 3600
                    await self.redis_client.expire(task_key, ttl_seconds)

                logger.debug(f"Task {task.id} saved to Redis successfully")

            except Exception as e:
                logger.error(f"Error saving task {task.id} to Redis: {e}")
                raise

    async def get(self, task_id: str, context: ServerCallContext | None = None) -> Task | None:
        """
        Retrieve a task from Redis by ID

        Args:
            task_id: The task ID to retrieve
            context: Optional gRPC server call context

        Returns:
            Task object if found, None otherwise
        """
        if not self._initialized:
            await self.initialize()

        async with self.lock:
            try:
                task_key = self._task_key(task_id)
                task_data = await self.redis_client.get(task_key)

                if not task_data:
                    logger.debug(f"Task {task_id} not found in Redis")
                    return None

                # Deserialize task from JSON
                task = Task.model_validate_json(task_data)
                logger.debug(f"Task {task_id} retrieved from Redis successfully")
                return task

            except Exception as e:
                logger.error(f"Error retrieving task {task_id} from Redis: {e}")
                raise

    async def delete(self, task_id: str, context: ServerCallContext | None = None) -> None:
        """
        Delete a task from Redis

        Args:
            task_id: The task ID to delete
            context: Optional gRPC server call context
        """
        if not self._initialized:
            await self.initialize()

        async with self.lock:
            try:
                task_key = self._task_key(task_id)
                meta_key = self._metadata_key(task_id)

                await self.redis_client.delete(task_key, meta_key)
                logger.debug(f"Task {task_id} deleted from Redis successfully")

            except Exception as e:
                logger.error(f"Error deleting task {task_id} from Redis: {e}")
                raise
