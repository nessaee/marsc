#!/bin/bash

# PlayerOne Camera SDK setup script

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
UDEV_RULES_SRC="$SCRIPT_DIR/lib/99-player_one_astronomy.rules"
UDEV_RULES_DEST="/etc/udev/rules.d/99-player_one_astronomy.rules"

# Check if script is run as root
if [ "$EUID" -ne 0 ]; then
  echo "This setup script needs to be run as root to install udev rules."
  echo "Please run: sudo $0"
  exit 1
fi

echo "Setting up PlayerOne Camera SDK..."

# Check if the source udev rules file exists
if [ ! -f "$UDEV_RULES_SRC" ]; then
  echo "ERROR: udev rules file not found at $UDEV_RULES_SRC"
  echo "Make sure you have extracted the SDK properly."
  exit 1
fi

# Install udev rules
echo "Installing udev rules for camera access..."
cp "$UDEV_RULES_SRC" "$UDEV_RULES_DEST"
if [ $? -ne 0 ]; then
  echo "ERROR: Failed to copy udev rules file to $UDEV_RULES_DEST"
  exit 1
fi

# Reload udev rules
echo "Reloading udev rules..."
udevadm control --reload-rules
udevadm trigger

echo "Done! You should now be able to access your PlayerOne camera without root privileges."
echo "If your camera is already connected, please disconnect and reconnect it."
echo "To use the camera, run: python $SCRIPT_DIR/playerone_mars.py"

exit 0