#!/bin/bash
set -euo pipefail

DATASETS=("dataset.npz")
MOMENT_FILES=("mom1.txt")

LOGDIR="./logs"
MAXJOBS=${SLURM_NTASKS:-1}

SEEDS=(1)
NTS=(250 500)
SIGMAS=(1.0 2.0)
NWINS=(8)

mkdir -p "$LOGDIR"

COMMANDS_FILE=$(mktemp)

for DATASET in "${DATASETS[@]}"; do
  DSTAG="$(basename "$DATASET" .npz)"

  for MOMENT_FILE in "${MOMENT_FILES[@]}"; do
    MOMTAG="$(basename "$MOMENT_FILE" .txt)"

    for SEED in "${SEEDS[@]}"; do
      for NT in "${NTS[@]}"; do
        for SIGMA in "${SIGMAS[@]}"; do
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
done

echo "Launching parallel runs with MAXJOBS=$MAXJOBS ..."

xargs -d '\n' -I CMD -P "$MAXJOBS" \
  srun --exclusive -N1 -n1 -c1 bash -lc "CMD" \
  < "$COMMANDS_FILE"

rm -f "$COMMANDS_FILE"

echo "All parallel runs finished."