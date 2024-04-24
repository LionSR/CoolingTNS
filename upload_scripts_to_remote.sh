#!/bin/bash

# rsync bash scripts to tqog01  
for dir in CoolingMPO.sh; do
    if [[ -d $dir ]]; then
        echo "Syncing $dir to tqo01 cluster"
        rsync -avz --include="*.sh" --exclude="*" "$dir" tqog01:/ptmp/mpq/siruilu/CoolingTNS/$dir
    fi
done


# Print the current time
current_time=$(date +"%H:%M:%S")
echo "Current time: $current_time"