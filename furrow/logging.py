import structlog
from .config import settings

def configure_logging() -> None:
    """Configure structlog with console renderer and log level from settings."""
    log_level = settings.log_level.upper()
    
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.dev.ConsoleRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Set the log level
    structlog.stdlib.reinit_defaults(log_level=log_level)

def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Get a structlog logger with the given name."""
    return structlog.get_logger(name)

# Configure logging on module import
configure_logging()