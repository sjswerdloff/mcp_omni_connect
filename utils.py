import logging
import platform
import sys
from pathlib import Path
import os
import colorlog 

# configuration setting for logging
LOGLEVEL = os.environ.get('LOGLEVEL', 'info').upper()
logger = logging.getLogger("MCPOmni Connect CLI")
logger.setLevel(LOGLEVEL)

# Create a color handler
handler = colorlog.StreamHandler()
formatter = colorlog.ColoredFormatter(
    '%(log_color)s%(levelname)s:   %(asctime)s - %(name)s -  %(message)s',
    log_colors={
        'DEBUG': 'cyan',
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'red,bg_white',
    }
)
handler.setFormatter(formatter)
logger.addHandler(handler)

def setup_platform():
    """Setup platform-specific configurations"""
    system = platform.system().lower()
    
    # Detect if running in WSL or Cygwin
    is_wsl = False
    is_cygwin = False
    
    if system == "linux":
        # Check for WSL
        try:
            with open('/proc/version', 'r') as f:
                if 'microsoft' in f.read().lower():
                    is_wsl = True
        except:
            pass
        # Check for Cygwin
        if 'CYGWIN' in platform.system():
            is_cygwin = True
    
    if system == "windows" or is_wsl or is_cygwin:
        # Windows/WSL/Cygwin-specific setup
        try:
            import colorama
            colorama.init()
            # Enable ANSI escape sequences
            if hasattr(sys.stdout, 'reconfigure'):
                sys.stdout.reconfigure(encoding='utf-8')
        except ImportError:
            logger.warning("Colorama not installed. Colors might not display correctly on Windows-like systems.")
    
    elif system == "darwin":
        # macOS-specific setup
        os.environ['LC_CTYPE'] = 'UTF-8'
    
    # Set up cross-platform config directory
    config_dir = Path.home() / ".mcp_omni_connect"
    config_dir.mkdir(exist_ok=True)
    
    return {
        "system": system,
        "config_dir": config_dir,
        "is_windows": system == "windows",
        "is_macos": system == "darwin",
        "is_linux": system == "linux",
        "is_wsl": is_wsl,
        "is_cygwin": is_cygwin,
        "is_windows_like": system == "windows" or is_wsl or is_cygwin
    }

def get_config_path(filename: str) -> Path:
    """Get the appropriate config file path for the current platform"""
    # Try user's config directory first
    user_config = Path.home() / ".mcp_omni_connect" / filename
    if user_config.exists():
        return user_config
    
    # Fall back to current directory
    local_config = Path(filename)
    if local_config.exists():
        return local_config
    
    # If neither exists, return the user config path as default
    return user_config

def ensure_unicode_paths(path_str: str) -> str:
    """Ensure paths are properly encoded for the current platform"""
    platform_info = setup_platform()
    
    if platform_info["is_windows_like"]:
        # Handle Windows-style paths, including WSL paths
        path = Path(path_str)
        if platform_info["is_wsl"]:
            # Convert Windows paths to WSL paths if needed
            if path_str.startswith(('/mnt/', '/')):
                return path_str
            return f"/mnt/c/{path_str.replace(':', '').replace('\\', '/')}"
        return str(path)
    return path_str