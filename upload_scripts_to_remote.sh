#!/bin/bash

# rsync only .sh files in the current directory to tqog01
echo "Syncing .sh files in the current directory to tqo01 cluster"
rsync -avz --include="*.sh" --exclude="*" ./ tqog01:/ptmp/mpq/siruilu/CoolingTNS/

# Print the current time
current_time=$(date +"%H:%M:%S")
echo "Current time: $current_time"