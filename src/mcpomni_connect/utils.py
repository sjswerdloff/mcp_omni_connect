import logging
import colorlog
import sys
from pathlib import Path
import uuid
import subprocess
import platform

# Configure logging
logger = logging.getLogger("mcpomni_connect")
logger.setLevel(logging.INFO)

# Remove any existing handlers
for handler in logger.handlers[:]:
    logger.removeHandler(handler)

# Create console handler with immediate flush
console_handler = colorlog.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)

# Create file handler with immediate flush
log_file = Path("mcpomni_connect.log")
file_handler = logging.FileHandler(log_file, mode='a')
file_handler.setLevel(logging.INFO)

# Create formatters
console_formatter = colorlog.ColoredFormatter(
    "%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    log_colors={
        'DEBUG': 'cyan',
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'red,bg_white',
    }
)

file_formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# Set formatters
console_handler.setFormatter(console_formatter)
file_handler.setFormatter(file_formatter)

# Add handlers
logger.addHandler(console_handler)
logger.addHandler(file_handler)

# Configure handlers to flush immediately
console_handler.flush = sys.stdout.flush
file_handler.flush = lambda: file_handler.stream.flush()

def get_mac_address() -> str:
    """Get the MAC address of the client machine.
    
    Returns:
        str: The MAC address as a string, or a fallback UUID if MAC address cannot be determined.
    """
    try:
        if platform.system() == "Linux":
            # Try to get MAC address from /sys/class/net/
            for interface in ["eth0", "wlan0", "en0"]:
                try:
                    with open(f"/sys/class/net/{interface}/address") as f:
                        mac = f.read().strip()
                        if mac:
                            return mac
                except FileNotFoundError:
                    continue
            
            # Fallback to using ip command
            result = subprocess.run(['ip', 'link', 'show'], capture_output=True, text=True)
            for line in result.stdout.split('\n'):
                if 'link/ether' in line:
                    return line.split('link/ether')[1].split()[0]
                    
        elif platform.system() == "Darwin":  # macOS
            result = subprocess.run(['ifconfig'], capture_output=True, text=True)
            for line in result.stdout.split('\n'):
                if 'ether' in line:
                    return line.split('ether')[1].split()[0]
                    
        elif platform.system() == "Windows":
            result = subprocess.run(['getmac'], capture_output=True, text=True)
            for line in result.stdout.split('\n'):
                if ':' in line and '-' in line:  # Look for MAC address format
                    return line.split()[0]
                    
    except Exception as e:
        logger.warning(f"Could not get MAC address: {e}")
        
    # If all else fails, generate a UUID
    return str(uuid.uuid4())

# Create a global instance of the MAC address
CLIENT_MAC_ADDRESS = get_mac_address()
