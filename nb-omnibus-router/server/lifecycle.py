"""
Server lifecycle management - startup and shutdown handlers
"""

import logging
from typing import Callable, List, Optional, Any
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

from utils.cache import RedisCache, CacheConfig, CacheWarmer, register_default_warming_tasks
from server.config import ServerConfig

logger = logging.getLogger(__name__)


@dataclass
class StartupTask:
    """Represents a startup task"""

    name: str
    callable: Callable
    priority: int = 100
    depends_on: List[str] = field(default_factory=list)


@dataclass
class ShutdownTask:
    """Represents a shutdown task"""

    name: str
    callable: Callable
    priority: int = 100


class ServerLifecycle:
    """
    Manages server startup and shutdown lifecycle
    """

    def __init__(self, config: ServerConfig):
        self.config = config
        self.startup_tasks: List[StartupTask] = []
        self.shutdown_tasks: List[ShutdownTask] = []
        self._cache: Optional[RedisCache] = None
        self._initialized = False

    def register_startup_task(
        self,
        name: str,
        callable: Callable,
        priority: int = 100,
        depends_on: Optional[List[str]] = None,
    ):
        """Register a startup task"""
        task = StartupTask(
            name=name, callable=callable, priority=priority, depends_on=depends_on or []
        )
        self.startup_tasks.append(task)
        # Sort by priority (lower = earlier)
        self.startup_tasks.sort(key=lambda t: t.priority)
        logger.debug(f"Registered startup task: {name} (priority: {priority})")

    def register_shutdown_task(self, name: str, callable: Callable, priority: int = 100):
        """Register a shutdown task"""
        task = ShutdownTask(name=name, callable=callable, priority=priority)
        self.shutdown_tasks.append(task)
        # Sort by priority (higher = earlier for shutdown)
        self.shutdown_tasks.sort(key=lambda t: -t.priority)
        logger.debug(f"Registered shutdown task: {name} (priority: {priority})")

    async def _initialize_cache(self):
        """Initialize Redis cache connection"""
        if not self.config.cache_enabled:
            logger.info("Cache disabled in configuration")
            return

        cache_config = CacheConfig(
            host=self.config.redis_host,
            port=self.config.redis_port,
            db=self.config.redis_db,
            password=self.config.redis_password,
            default_ttl=self.config.cache_ttl,
        )

        self._cache = RedisCache(cache_config)
        connected = await self._cache.connect()

        if connected:
            logger.info("✓ Redis cache connected successfully")

            # Warm up cache
            warmer = CacheWarmer(self._cache)
            register_default_warming_tasks(warmer)
            warmup_results = await warmer.warm()
            logger.info(f"✓ Cache warming completed: {warmup_results}")
        else:
            logger.warning("⚠ Redis cache unavailable, using local fallback")

    async def _close_cache(self):
        """Close cache connection"""
        if self._cache:
            await self._cache.disconnect()
            logger.info("✓ Cache connection closed")

    def setup_default_tasks(self):
        """Setup default startup/shutdown tasks"""
        # Register cache initialization
        self.register_startup_task(
            name="cache_init",
            callable=self._initialize_cache,
            priority=10,  # High priority - initialize early
        )

        # Register cache cleanup
        self.register_shutdown_task(name="cache_cleanup", callable=self._close_cache, priority=10)

    async def on_startup(self):
        """Execute all startup tasks"""
        logger.info("=" * 50)
        logger.info("Starting NB Omnibus Server...")
        logger.info(f"Environment: {self.config.environment}")
        logger.info(f"Debug mode: {self.config.debug}")
        logger.info("=" * 50)

        completed_tasks = set()

        for task in self.startup_tasks:
            try:
                # Check dependencies
                for dep in task.depends_on:
                    if dep not in completed_tasks:
                        logger.warning(f"Task {task.name} waiting for dependency: {dep}")

                logger.info(f"Executing startup task: {task.name}")

                # Execute task
                if hasattr(task.callable, "__call__"):
                    if hasattr(task.callable, "__await__"):
                        await task.callable()
                    else:
                        task.callable()

                completed_tasks.add(task.name)
                logger.info(f"✓ Startup task completed: {task.name}")

            except Exception as e:
                logger.error(f"✗ Startup task failed: {task.name} - {e}")
                if not self.config.debug:
                    raise

        self._initialized = True
        logger.info("=" * 50)
        logger.info("✓ Server startup completed successfully")
        logger.info("=" * 50)

    async def on_shutdown(self):
        """Execute all shutdown tasks"""
        logger.info("=" * 50)
        logger.info("Shutting down NB Omnibus Server...")
        logger.info("=" * 50)

        for task in self.shutdown_tasks:
            try:
                logger.info(f"Executing shutdown task: {task.name}")

                if hasattr(task.callable, "__call__"):
                    if hasattr(task.callable, "__await__"):
                        await task.callable()
                    else:
                        task.callable()

                logger.info(f"✓ Shutdown task completed: {task.name}")

            except Exception as e:
                logger.error(f"✗ Shutdown task failed: {task.name} - {e}")

        self._initialized = False
        logger.info("=" * 50)
        logger.info("✓ Server shutdown completed")
        logger.info("=" * 50)

    def get_cache(self) -> Optional[RedisCache]:
        """Get the cache instance"""
        return self._cache

    @property
    def is_initialized(self) -> bool:
        """Check if server is initialized"""
        return self._initialized
