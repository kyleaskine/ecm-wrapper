"""
Jinja2 template configuration module.
Provides a singleton templates instance that can be imported across the app.
"""
from fastapi.templating import Jinja2Templates
from pathlib import Path

from .utils.html_helpers import (
    esc,
    format_composite_display,
    calculate_t_level_progress,
    get_health_class,
    format_status_badge,
    format_work_status_class,
    format_time_remaining,
    format_factors_display
)

# Initialize Jinja2 templates
template_dir = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(template_dir))

# Add custom filters and globals to Jinja2
templates.env.globals.update({
    'esc': esc,
    'format_composite': format_composite_display,
    'calc_t_progress': calculate_t_level_progress,
    'get_health_class': get_health_class,
    'format_status': format_status_badge,
    'format_work_status': format_work_status_class,
    'format_time': format_time_remaining,
    'format_factors': format_factors_display
})
