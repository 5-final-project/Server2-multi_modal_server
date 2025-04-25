from celery import Celery
from .config import settings
import logging

logger = logging.getLogger(__name__)

# Initialize Celery
# The first argument is the name of the current module, important for Celery's auto-discovery
# The second argument 'broker' specifies the URL of the message broker (Redis)
# The third argument 'backend' specifies the URL of the result backend (also Redis)
# 'include' tells Celery which modules contain tasks
celery_app = Celery(
    "inference_worker",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
    include=["app.core.inference_worker"] # Point to the module where the task will be defined
)

# Optional Celery configuration settings
celery_app.conf.update(
    task_serializer="json",        # Use json for task serialization
    result_serializer="json",      # Use json for result serialization
    accept_content=["json"],       # Accept json content
    timezone="UTC",                # Use UTC timezone
    enable_utc=True,
    # Optional: Configure task tracking and states
    task_track_started=True,
    # Optional: Set concurrency limits if needed (handled by worker command usually)
    # worker_concurrency=4,
)

logger.info(f"Celery app initialized with broker: {settings.CELERY_BROKER_URL} and backend: {settings.CELERY_RESULT_BACKEND}")

# Example: You can define shared tasks or configurations here if needed
# @celery_app.task(bind=True)
# def debug_task(self):
#     print(f'Request: {self.request!r}')
