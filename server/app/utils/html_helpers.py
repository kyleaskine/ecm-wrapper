"""
HTML rendering helper functions for template rendering.
Provides shared utility functions for formatting data for display.
"""
from typing import Dict, Any, Optional
import html as html_module


def esc(text: Any) -> str:
    """
    Escape HTML special characters to prevent XSS attacks.

    Args:
        text: Text to escape (will be converted to string)

    Returns:
        HTML-escaped string
    """
    return html_module.escape(str(text))


def format_composite_display(number: str, max_length: int = 50) -> str:
    """
    Format composite number for display, truncating if too long.

    Args:
        number: Composite number string
        max_length: Maximum length before truncation

    Returns:
        Formatted number string with ellipsis if truncated
    """
    if len(number) > max_length:
        return number[:max_length] + '...'
    return number


def calculate_t_level_progress(current_t: Optional[float],
                               target_t: Optional[float]) -> Dict[str, Any]:
    """
    Calculate t-level progress metrics for display.

    Args:
        current_t: Current t-level value
        target_t: Target t-level value

    Returns:
        Dictionary with progress percentage and color
    """
    current = current_t or 0.0
    target = target_t or 0.0

    if target > 0:
        progress_pct = (current / target) * 100
    else:
        progress_pct = 0

    # Determine color based on progress
    if progress_pct >= 100:
        color = "#198754"  # Green
    elif progress_pct >= 50:
        color = "#fd7e14"  # Orange
    else:
        color = "#dc3545"  # Red

    return {
        'current_t': current,
        'target_t': target,
        'progress_pct': progress_pct,
        'progress_color': color
    }


def get_health_class(percentage: float,
                     good_threshold: float = 80,
                     warning_threshold: float = 50) -> str:
    """
    Get CSS class for health indicator based on percentage.

    Args:
        percentage: Health percentage value
        good_threshold: Minimum percentage for 'good' status
        warning_threshold: Minimum percentage for 'warning' status

    Returns:
        CSS class name: 'health-good', 'health-warning', or 'health-critical'
    """
    if percentage >= good_threshold:
        return 'health-good'
    if percentage >= warning_threshold:
        return 'health-warning'
    return 'health-critical'


def format_status_badge(is_prime: bool, is_fully_factored: bool) -> Dict[str, str]:
    """
    Get status badge information for a composite.

    Args:
        is_prime: Whether the number is prime
        is_fully_factored: Whether the number is fully factored

    Returns:
        Dictionary with badge class and text
    """
    if is_prime:
        return {
            'class': 'status-complete',
            'text': 'Prime'
        }
    if is_fully_factored:
        return {
            'class': 'status-complete',
            'text': 'Fully Factored'
        }
    return {
        'class': 'status-pending',
        'text': 'Composite'
    }


def format_work_status_class(status: str) -> str:
    """
    Get CSS class for work assignment status.

    Args:
        status: Work assignment status string

    Returns:
        CSS class name for the status
    """
    return f"status-{status.replace('_', '-')}"


def format_time_remaining(expires_at, current_time) -> str:
    """
    Format time remaining until expiration.

    Args:
        expires_at: Expiration datetime
        current_time: Current datetime

    Returns:
        Formatted time string (e.g., "45m", "Expired")
    """
    if expires_at > current_time:
        time_remaining = expires_at - current_time
        minutes = int(time_remaining.total_seconds() / 60)
        return f"{minutes}m"
    return "Expired"


def format_factors_display(factors: list) -> str:
    """
    Format list of factors for HTML display.

    Args:
        factors: List of factor objects or strings

    Returns:
        HTML string with formatted factors
    """
    if not factors:
        return "None"

    factor_strs = [f'<span class="factor">{esc(f.factor if hasattr(f, "factor") else f)}</span>'
                   for f in factors]
    return " × ".join(factor_strs)


def format_number_with_commas(number: int) -> str:
    """
    Format a number with thousands separators.

    Args:
        number: Integer to format

    Returns:
        Formatted number string (e.g., "1,234,567")
    """
    return f"{number:,}"


def format_datetime(dt, format_string: str = '%Y-%m-%d %H:%M') -> str:
    """
    Format a datetime object to string.

    Args:
        dt: Datetime object or None
        format_string: strftime format string

    Returns:
        Formatted datetime string or 'Unknown'
    """
    if dt is None:
        return 'Unknown'
    return dt.strftime(format_string)


def truncate_text(text: str, max_length: int = 50, suffix: str = '...') -> str:
    """
    Truncate text to a maximum length with suffix.

    Args:
        text: Text to truncate
        max_length: Maximum length before truncation
        suffix: Suffix to add when truncated

    Returns:
        Truncated text with suffix if needed
    """
    if len(text) <= max_length:
        return text
    return text[:max_length] + suffix


def format_percentage(value: float, decimals: int = 1) -> str:
    """
    Format a float as a percentage string.

    Args:
        value: Float value (e.g., 75.5)
        decimals: Number of decimal places

    Returns:
        Formatted percentage string (e.g., "75.5%")
    """
    return f"{value:.{decimals}f}%"


def format_execution_time(seconds: float) -> str:
    """
    Format execution time in a human-readable way.

    Args:
        seconds: Execution time in seconds

    Returns:
        Formatted time string (e.g., "2h 30m" or "45s")
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        minutes = int(seconds / 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.0f}s"
    hours = int(seconds / 3600)
    minutes = int((seconds % 3600) / 60)
    return f"{hours}h {minutes}m"


def get_priority_display(priority: int) -> Dict[str, Any]:
    """
    Get display information for a priority value.

    Args:
        priority: Priority integer

    Returns:
        Dictionary with color and formatted text
    """
    if priority > 0:
        return {
            'value': priority,
            'color': '#198754',
            'text': f'+{priority}'
        }
    if priority < 0:
        return {
            'value': priority,
            'color': '#dc3545',
            'text': str(priority)
        }
    return {
        'value': priority,
        'color': '#666',
        'text': '0'
    }


def get_unauthorized_redirect_html() -> str:
    """
    Get HTML for unauthorized access redirect page.

    Returns:
        HTML string that redirects to login page
    """
    return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Unauthorized</title>
            <script>
                sessionStorage.removeItem('admin_api_key');
                window.location.href = '/api/v1/admin/login';
            </script>
        </head>
        <body>
            <p>Redirecting to login...</p>
        </body>
        </html>
        """
