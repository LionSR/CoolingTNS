#!/bin/bash

while true; do
  # Sync directories from tqog01
  rsync -avzu --progress tqog01:/ptmp/mpq/siruilu/CoolingTNS/Results .

  # Print the current time
  current_time=$(date +"%H:%M:%S")
  echo "Current time: $current_time"

  # Wait for 1 hour before syncing again
  sleep 3600
done
