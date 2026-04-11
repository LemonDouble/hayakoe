"""공유 Rich 콘솔 인스턴스 및 테마."""

from rich.console import Console
from rich.theme import Theme


theme = Theme({
    "info": "cyan",
    "success": "bold green",
    "warning": "bold yellow",
    "error": "bold red",
    "step": "bold magenta",
    "header": "bold bright_white",
    "dim": "dim",
    "accent": "bold cyan",
    "label": "dim cyan",
    "value": "bright_white",
    "muted": "bright_black",
})

console = Console(theme=theme)


LOGO = r"""[accent]
  ╦ ╦╔═╗╦ ╦╔═╗╦╔═╔═╗╔═╗
  ╠═╣╠═╣╚╦╝╠═╣╠╩╗║ ║║╣
  ╩ ╩╩ ╩ ╩ ╩ ╩╩ ╩╚═╝╚═╝[/accent] [dim]Dev Tools[/dim]
"""
