# utils/celery_utils/system_tasks.py

"""Responsible for all periodic, system-wide maintenance and monitoring."""

# ===== STANDARD LIBRARY IMPORTS =====
import os
import time
import tempfile  # FIX: Added missing import for tempfile
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Iterator, Optional

# ===== THIRD-PARTY IMPORTS =====
from celery.schedules import crontab
from celery.utils.log import get_task_logger  # FIX: Added missing import for logger
from dotenv import load_dotenv  # FIX: Added missing import for dotenv
import psutil

# ===== PROJECT MODULES =====
from tasks.celery_app import celery_app
from utils.supabase_utils import supabase_client

# FIX: Import the global instances that your tasks depend on.
# You need to replace 'path.to.your.modules' with the actual paths in your project.
# from utils.metrics import metrics_collector
# from utils.document_loaders.performance import batch_processor

# ——— Logging & Env Load ———————————————————————————————————————————————————————————
load_dotenv()
logger = get_task_logger(__name__)
# logger.propagate = False # This can sometimes hide logs, consider keeping it True during dev

# ——— Constants ———————————————————————————————————————————————————————————————————
# FIX: Defined queue names as constants instead of using the large Enum
INGEST_QUEUE = 'ingest'
PARSE_QUEUE = 'parsing'
EMBED_QUEUE = 'embedding'
FINAL_QUEUE = 'finalize'
MONITORING_QUEUE = 'monitoring'
MAINTENANCE_QUEUE = 'maintenance'
OPTIMIZATION_QUEUE = 'optimization'


# ——— System Health & Maintenance Tasks ———————————————————————————————————————————————

@celery_app.task(bind=True, queue=MONITORING_QUEUE)
def system_health_check(self) -> Dict[str, Any]:
    """
    Comprehensive system health monitoring.
    """
    try:
        # System resource metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')

        # Queue depth monitoring
        queue_stats = {}
        # FIX: Used the string constants defined above
        for queue_name in [INGEST_QUEUE, PARSE_QUEUE, EMBED_QUEUE, FINAL_QUEUE]:
            try:
                inspect = self.app.control.inspect()
                # Correctly inspect active tasks for a specific queue
                active_tasks = inspect.active()
                queue_length = 0
                if active_tasks:
                    # Celery returns a dict of {worker_name: [list_of_tasks]}
                    for worker, tasks in active_tasks.items():
                         if worker.endswith(f'.{queue_name}'): # Check if worker is for the queue
                            queue_length += len(tasks)
                queue_stats[queue_name] = queue_length
            except Exception as e:
                logger.warning(f"Could not inspect queue '{queue_name}': {e}")
                queue_stats[queue_name] = -1  # Unable to determine

        # Health assessment
        health_status = "healthy"
        issues = []
        if cpu_percent > 85:
            health_status = "degraded"
            issues.append(f"High CPU usage: {cpu_percent:.1f}%")
        if memory.percent > 90:
            health_status = "critical"
            issues.append(f"High memory usage: {memory.percent:.1f}%")

        # Generate scaling recommendations
        recommendations = []
        total_queue_depth = sum(v for v in queue_stats.values() if v > 0)
        if total_queue_depth > 100:
            recommendations.append("Consider scaling up worker instances")
        if cpu_percent < 30 and total_queue_depth < 10:
            recommendations.append("System is under-utilized, consider scaling down")

        health_report = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'status': health_status,
            'issues': issues,
            'recommendations': recommendations,
            'system_metrics': {
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'disk_percent': disk.percent,
            },
            'queue_stats': queue_stats,
        }
        supabase_client.table('system_health_reports').insert(health_report).execute()
        return health_report

    except Exception as e:
        logger.error(f"Health check failed: {e}", exc_info=True)
        return {'status': 'error', 'error': str(e)}


@celery_app.task(bind=True, queue=MAINTENANCE_QUEUE)
def cleanup_orphaned_resources(self) -> Dict[str, Any]:
    """
    Automated cleanup of orphaned resources like stale files and records.
    """
    cleanup_stats = {'temp_files_removed': 0, 'stale_records_cleaned': 0, 'old_metrics_purged': 0}
    
    # 1. Clean up temporary files older than 24 hours
    try:
        temp_dir = tempfile.gettempdir()
        cutoff_time = time.time() - (24 * 3600)
        for filename in os.listdir(temp_dir):
            filepath = os.path.join(temp_dir, filename)
            if os.path.isfile(filepath) and os.path.getmtime(filepath) < cutoff_time:
                os.unlink(filepath)
                cleanup_stats['temp_files_removed'] += 1
    except Exception as e:
        logger.error(f"Error cleaning temp files: {e}")

    # 2. Clean up documents stuck in processing states for > 4 hours
    try:
        four_hours_ago = (datetime.now(timezone.utc) - timedelta(hours=4)).isoformat()
        # FIX: Replaced Enum with simple strings for statuses to avoid importing the large Enum
        stale_statuses = ['PARSING', 'EMBEDDING', 'CHUNKING']
        
        stale_docs_res = supabase_client.table('document_sources').select('id').in_('vector_embed_status', stale_statuses).lt('created_at', four_hours_ago).execute()
        
        if stale_docs_res.data:
            doc_ids = [doc['id'] for doc in stale_docs_res.data]
            supabase_client.table('document_sources').update({
                'vector_embed_status': 'FAILED_PARSING', # A generic failure state
                'error_message': 'Processing timed out - cleaned by maintenance task'
            }).in_('id', doc_ids).execute()
            cleanup_stats['stale_records_cleaned'] = len(doc_ids)
    except Exception as e:
        logger.error(f"Error cleaning stale DB records: {e}")

    logger.info(f"Cleanup completed: {cleanup_stats}")
    return cleanup_stats


# ——— Production Deployment & Initialization ————————————————————————————————————————

def initialize_production_pipeline():
    """
    Configures the Celery Beat schedule for periodic system tasks.
    This function should be called once from `celery_app.py`.
    """
    # FIX: Removed readiness check from here; it's better as a separate script or startup check
    # FIX: Removed unused global variable declarations
    
    logger.info("Scheduling periodic maintenance and monitoring tasks...")
    
    celery_app.conf.beat_schedule = {
        'system-health-check-5m': {
            'task': 'tasks.system_tasks.system_health_check', # Use full path to task
            'schedule': 300.0,  # Every 5 minutes
        },
        'cleanup-orphaned-resources-daily': {
            'task': 'tasks.system_tasks.cleanup_orphaned_resources', # Use full path
            'schedule': crontab(hour=2, minute=30),  # Daily at 2:30 AM UTC
        },
        # 'optimize-performance': {
        #     'task': 'tasks.system_tasks.optimize_embedding_performance',
        #     'schedule': crontab(hour='*/1'),  # Every hour
        # }
    }
    
    # Optional: Add a check for a specific environment variable to enable the beat schedule
    # if os.getenv('ENABLE_CELERY_BEAT', 'false').lower() == 'true':
    #    ...
    
    logger.info("Celery Beat schedule configured.")
    return True


def validate_production_readiness() -> Dict[str, Any]:
    """
    Validate that all production requirements are met
    """
    checks = {
        'environment_variables': True,
        'database_connectivity': True,
        'external_services': True,
        'resource_limits': True,
        'monitoring_setup': True
    }
    
    issues = []
    
    # Check environment variables
    required_vars = ['OPENAI_API_KEY', 'SUPABASE_URL', 'SUPABASE_KEY', 'AWS_ACCESS_KEY_ID']
    for var in required_vars:
        if not os.getenv(var):
            checks['environment_variables'] = False
            issues.append(f"Missing environment variable: {var}")
    
    # Check database connectivity
    try:
        supabase_client.table('document_sources').select('id').limit(1).execute()
    except Exception as e:
        checks['database_connectivity'] = False
        issues.append(f"Database connectivity failed: {e}")
    
    # Check resource limits
    memory_gb = psutil.virtual_memory().total / (1024**3)
    if memory_gb < 2:
        checks['resource_limits'] = False
        issues.append(f"Insufficient memory: {memory_gb:.1f}GB < 2GB minimum")
    
    return {
        'ready_for_production': all(checks.values()),
        'checks': checks,
        'issues': issues,
        'recommendations': [
            "Set up comprehensive monitoring and alerting",
            "Configure auto-scaling based on queue depth",
            "Implement proper backup and disaster recovery",
            "Set up log aggregation and analysis"
        ]
    }