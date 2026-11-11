#!/usr/bin/env bash
# Full 16x16 cross-quarter validation (every model quarter x every imagery quarter, 2016-2019)
#
# Usage
#   bash run_cross_quarter_full.sh            # defaults to oh / wealth_index
#   bash run_cross_quarter_full.sh ga wealth_index
#
# The diagonal (model evaluated on its own quarter) is skipped by default because the
# matrix uses each model's held-out test set there. Set SKIP_SELF=0 to include it.

set -uo pipefail

STATE="${1:-ca}"
VARIABLE="${2:-wealth_index}"
VERSION="v1"
YEARS=(2016 2017 2018 2019)
QUARTERS=(1 2 3 4)
SKIP_SELF=1

PROJECT_DIR="/home/hbaier/projects/tlags_v2"
CONFIG_DIR="${PROJECT_DIR}/configs/sail/tlags/${STATE}/cross"

cd "$PROJECT_DIR" || { echo "Could not cd to $PROJECT_DIR"; exit 1; }

FAILED=()

for MODEL_YEAR in "${YEARS[@]}"; do
  for MODEL_Q in "${QUARTERS[@]}"; do
    for IMG_YEAR in "${YEARS[@]}"; do

      # Build the list of imagery quarters for this model x imagery-year block
      IMG_QS=()
      for IMG_Q in "${QUARTERS[@]}"; do
        if [[ "$SKIP_SELF" -eq 1 && "$MODEL_YEAR" == "$IMG_YEAR" && "$MODEL_Q" == "$IMG_Q" ]]; then
          continue
        fi
        IMG_QS+=("${IMG_YEAR}Q${IMG_Q}")
      done

      echo "=== ${STATE} model Q${MODEL_Q} ${MODEL_YEAR} on ${IMG_YEAR} imagery (${IMG_QS[*]}) ==="

      python pipeline_configs/generate_cross_validate_config.py \
        --state "$STATE" --variable "$VARIABLE" \
        --model-year "$MODEL_YEAR" --model-quarter "$MODEL_Q" \
        --imagery-quarters "${IMG_QS[@]}" \
        || { echo "Config generation failed for Q${MODEL_Q} ${MODEL_YEAR} on ${IMG_YEAR}"; FAILED+=("gen_q${MODEL_Q}_${MODEL_YEAR}_on_${IMG_YEAR}"); continue; }

      for IQ in "${IMG_QS[@]}"; do
        IY="${IQ%Q*}"
        IQN="${IQ#*Q}"
        CFG="${CONFIG_DIR}/${STATE}_q${MODEL_Q}_${MODEL_YEAR}_${VARIABLE}_${VERSION}_on_${IY}q${IQN}_validate.yml"
        simba "$CFG" || FAILED+=("$CFG")
      done

    done
  done
done

# Build the full 16x16 matrix
ALL_QS=()
for Y in "${YEARS[@]}"; do
  for Q in "${QUARTERS[@]}"; do
    ALL_QS+=("${Y}Q${Q}")
  done
done

python 3_analysis/build_cross_quarter_r2_matrix.py \
  --state "$STATE" --variable "$VARIABLE" \
  --quarters "${ALL_QS[@]}"

if [[ ${#FAILED[@]} -gt 0 ]]; then
  echo ""
  echo "Failed runs (${#FAILED[@]})"
  printf '  %s\n' "${FAILED[@]}"
fi
