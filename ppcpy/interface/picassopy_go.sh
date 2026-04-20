#!/usr/bin/env bash
#=====================================================================

set -euo pipefail   # safety: exit on error, treat undefined vars as errors

# ---------- default values ----------
BASE_DIR="/data/level0/polly"
MERGE_SINGLE=0   # false
TIMESTAMP=""
DEVICE=""
PICASSO_CFG=""

# ---------- helper functions ----------
usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Options:
  --startdate YYYYMMDD         StartDate of measurement (required)
  --enddate YYYYMMDD           EndDate of measurement (required)
  --device DEVICE_NAME         Name of the Polly device (required)
  --base_dir DIR               Base directory of level‑0 data (default: $BASE_DIR)
  --picasso_config_file FILE   Picasso JSON config file (required)
  --merge_to_single_24h_file   Merge all level‑0 files of the day into one 24‑h file
  -h, --help                   Show this help and exit
EOF
    exit 1
}

log_error()   { printf "%s [ERROR] %s\n" "$(date +"%Y-%m-%d %H:%M:%S,%3N")" "$*" >&2; }
log_info()    { printf "%s [INFO]  %s\n" "$(date +"%Y-%m-%d %H:%M:%S,%3N")" "$*"; }
log_debug()   { printf "%s [DEBUG] %s\n" "$(date +"%Y-%m-%d %H:%M:%S,%3N")" "$*"; }

# ---------- argument parsing ----------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --startdate)
            STARTDATE="${2:?Missing argument for --startdate}"
            shift 2
            ;;
        --enddate)
            ENDDATE="${2:?Missing argument for --enddate}"
            shift 2
            ;;
        --device)
            DEVICE="${2:?Missing argument for --device}"
            shift 2
            ;;
        --base_dir)
            BASE_DIR="${2:?Missing argument for --base_dir}"
            shift 2
            ;;
        --picasso_config_file)
            PICASSO_CFG="${2:?Missing argument for --picasso_config_file}"
            shift 2
            ;;
        --merge_to_single_24h_file)
            MERGE_SINGLE=1
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            log_error "Unknown option: $1"
            usage
            ;;
    esac
done

# ---------- validation ----------
if [[ -z "$STARTDATE" ]]; then
    log_error "No start-timestamp specified. Aborting."
    exit 1
fi
if [[ -z "$ENDDATE" ]]; then
    log_error "No end-timestamp specified. Aborting."
    exit 1
fi

if [[ -z "$DEVICE" ]]; then
    log_error "No device specified. Aborting."
    exit 1
fi

if [[ -z "$PICASSO_CFG" ]]; then
    log_error "No picasso config file specified. Aborting."
    exit 1
fi

if [[ ! -f "$PICASSO_CFG" ]]; then
    log_error "Picasso config file not found: $PICASSO_CFG"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# three '..' → three parents up
if command -v realpath >/dev/null 2>&1; then
    ROOT_DIR="$(realpath "$SCRIPT_DIR/../..")"
else
    ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
fi

# Optional: expose it for debugging
printf '[DEBUG] ROOT_DIR = %s\n' "$ROOT_DIR"

# ---------- load picasso config parameters ----------

FILEINFO_NEW=$(jq -r '.fileinfo_new // empty' "$PICASSO_CFG")
PYTHON_PATH=$(jq -r '.pyBinDir // empty' "$PICASSO_CFG")
if [[ -z "$FILEINFO_NEW" || "$FILEINFO_NEW" == "null" ]]; then
    log_error "\"fileinfo_new\" not found (or empty) in $PICASSO_CFG"
    exit 1
fi

# Build the output path ─ same logic as:
OUTPUT_ROOT=$(dirname "$FILEINFO_NEW")          # parent directory
OUTPUT_PATH="${OUTPUT_ROOT%/}/${DEVICE}"        # ensure no double slash

log_info "Output path will be: $OUTPUT_PATH"
mkdir -p "$OUTPUT_PATH"


# ------------------------------------------------------------
# Bash helper that invokes python functions get_pollyxt_files and concat_files via python -c
# ------------------------------------------------------------
call_get_files() {
    local ts="$1"
    local dev="$2"
    local raw_folder="$3"
    local out_path="$4"

    "$PYTHON_PATH"/python3 -c "
import sys
from ppcpy.misc.concat import get_pollyxt_files

ts, dev, raw, out = '$ts', '$dev', '$raw_folder', '$out_path'

#  Call the function.
result = get_pollyxt_files(timestamp=ts,
                     device=dev,
                     raw_folder=raw,
                     output_path=out)

#  Print the result – Bash captures STDOUT.
if result is not None:
    for p in result:
        print(p)
"
}

call_concat_files() {
    local ts="$1"
    local dev="$2"
    local raw_folder="$3"
    local out_path="$4"

    "$PYTHON_PATH"/python3 -c "
import sys
from ppcpy.misc.concat import concat_files

ts, dev, raw, out = '$ts', '$dev', '$raw_folder', '$out_path'

#  Call the function.
result = concat_files(timestamp=ts,
                     device=dev,
                     raw_folder=raw,
                     output_path=out)

#  Print the result – Bash captures STDOUT.
if result:
    print(str(result).strip())
"
}

create_date_ls() {
## create DATE_LS
    dates=()
    for (( date=STARTDATE; date <= ENDDATE; )); do
        dates+=( "$date" )
        date="$(date --date="$date +1 days" +'%Y%m%d')"
    done
    
    for i in ${dates[@]}; do
    	YYYY=${i:0:4}
    	MM=${i:4:2}
    	DD=${i:6:2}
    	YYYYMMDD=$YYYY$MM$DD
    	DATE_LS+=( "$YYYYMMDD" )
    done
}


create_date_ls

# ---------- collect raw file names ----------
RAW_FILES=()   # Bash array
MERGE_SUCCESS=0
for TIMESTAMP in ${DATE_LS[@]}; do
    
    if (( MERGE_SINGLE )); then
        log_info "Merging raw files for $TIMESTAMP / $DEVICE into a single 24‑h file ..."
        ## grep to get rid of empty lines
        MERGED_FILE=$(call_concat_files "$TIMESTAMP" "$DEVICE" "$BASE_DIR" "$OUTPUT_PATH" | grep -v '^[[:space:]]*$') \
            || { echo "[ERROR] concat_files failed" >&2; MERGE_SUCCESS=0; }
    
        if [[ -z "$MERGED_FILE" ]]; then
            log_error "concat_files.py did not return a file name."
            MERGE_SUCCESS=0
            
        else
            RAW_FILES+=("$MERGED_FILE")
            MERGE_SUCCESS=1
        fi
    else
        # Call the Python helper that lists the individual level‑0 files.
        log_info "Fetching list of level‑0 files for $TIMESTAMP / $DEVICE ..."
       # mapfile -t RAW_FILES < <(python3 get_pollyxt_files.py \
        mapfile -t RAW_FILES < <(call_get_files "$TIMESTAMP" "$DEVICE" "$BASE_DIR" "$OUTPUT_PATH") \
    		|| { log_error "Failed to run get_pollyxt_files.py"; exit 1; }
    fi
    
    # ---------- sanity check ----------
    if (( ${#RAW_FILES[@]} == 0 )); then
        log_error "No files to process. Aborting."
        exit 1
    fi
    
    # ---------- processing loop ----------
    for rawfile in "${RAW_FILES[@]}"; do
        echo "$rawfile"
        # Guard against empty strings that might have slipped in.
        if [[ -z "$rawfile" ]]; then
            log_error "Encountered an empty file name – aborting."
            exit 1
        fi
    
        # verify that the file exists before processing.
        if [[ ! -f "$rawfile" ]]; then
            log_error "File not found: $rawfile"
            exit 1
        fi
    
        log_info "Processing file: $rawfile"
    
        # -----------------------------------------------------------------
        # Call the actual processing script.
        # -----------------------------------------------------------------
        
        PICASSO_OP_SCRIPT="${ROOT_DIR}/ppcpy/interface/picassopy_operational.py"
        "$PYTHON_PATH"/python3 "$PICASSO_OP_SCRIPT" \
            --date "$TIMESTAMP" \
            --device "$DEVICE"\
            --base_dir "$BASE_DIR"\
            --picasso_config_file "$PICASSO_CFG" \
            --level0_file_to_process "$rawfile" \
            || {
                log_error "Processing of $rawfile failed."
                exit 1
            }
    done
done

log_info "All files processed successfully."
exit 0
