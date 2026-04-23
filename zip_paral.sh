#!/bin/bash
set -euo pipefail

DATASETS=("dataset_bur_1.npz")
MOMENT_FILES=("mom2.txt")

LOGDIR="./logs"
MAXJOBS=${SLURM_NTASKS:-1}

SEEDS=(1)
NTS=(100 500 2500 5000 10000)
SIGMAS=(0.3 0.3 0.3 0.3 0.3)

NWINS=(1)  

mkdir -p "$LOGDIR"
COMMANDS_FILE=$(mktemp)

if [ "${#NTS[@]}" -ne "${#SIGMAS[@]}" ]; then
  echo "Error: NTS and SIGMAS must have the same length" >&2
  exit 1
fi

for DATASET in "${DATASETS[@]}"; do
  DSTAG="$(basename "$DATASET" .npz)"

  for MOMENT_FILE in "${MOMENT_FILES[@]}"; do
    MOMTAG="$(basename "$MOMENT_FILE" .txt)"

    for SEED in "${SEEDS[@]}"; do
      for i in "${!NTS[@]}"; do
        NT="${NTS[$i]}"
        SIGMA="${SIGMAS[$i]}"

        for NWIN in "${NWINS[@]}"; do 

          LOGFILE="${LOGDIR}/${DSTAG}_${MOMTAG}_seed-${SEED}_Nt-${NT}_sigma-${SIGMA}_nwin-${NWIN}.log"

          echo "python3 run_mgd.py \
            \"$SIGMA\" \
            \"$NT\" \
            \"$SEED\" \
            \"$DATASET\" \
            \"$MOMENT_FILE\" \
            \"$NWIN\" \
            > \"$LOGFILE\" 2>&1" >> "$COMMANDS_FILE"

        done
      done
    done
  done
done

echo "Launching parallel runs with MAXJOBS=$MAXJOBS ..."

xargs -d '\n' -I CMD -P "$MAXJOBS" \
  srun --exclusive -N1 -n1 -c1 bash -lc "CMD" \
  < "$COMMANDS_FILE"

rm -f "$COMMANDS_FILE"

echo "All parallel runs finished."