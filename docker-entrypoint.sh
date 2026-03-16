#!/bin/sh
set -eu

# Ensure necessary directories exist
mkdir -p /app/data /app/models /app/data/uploads

# Fix ownership for mounted volumes so `appuser` can write
chown -R appuser:appuser /app/data /app/models /app || true

# If the first arg looks like an option, prepend the default command
if [ "${1#-}" != "$1" ]; then
  set -- python "$@"
fi

# Exec the command as appuser
exec su -s /bin/sh appuser -c "$*"
