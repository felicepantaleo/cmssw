#!/usr/bin/env bash

# Exit the script immediately if any command fails
set -e

# Enable pipefail to propagate the exit status of the entire pipeline
set -o pipefail

############################
# Global configuration
############################

FOLDER_FILES="${FOLDER_FILES:-/data/${USER}/}"
DATASET="/RelValTTbar_14TeV/CMSSW_20_0_0_pre1-PU_150X_mcRun4_realistic_v1_STD_D121_RegeneratedGS_PU-v1/GEN-SIM-DIGI-RAW"

EVENTS=${EVENTS:-1000}
THREADS=4

# Benchmark geometry and NUMA slot. Defaults reproduce the cms-bot configuration
# (8 jobs x 16 threads x 16 streams on a 4-NUMA-node machine); override on hosts
# with a different topology or to shrink the footprint.
BENCH_JOBS=${BENCH_JOBS:-8}
BENCH_THREADS=${BENCH_THREADS:-16}
BENCH_STREAMS=${BENCH_STREAMS:-16}
BENCH_SLOT=${BENCH_SLOT:-"numa=0-3:mem=0-3"}

############################
# GPU Monitoring config
############################

ENABLE_RESOURCES_MONITORING=true
MONITOR_INTERVAL=1

# Check dependencies
if [[ "$ENABLE_RESOURCES_MONITORING" = true ]]; then
    if ! command -v nvidia-smi &>/dev/null; then
	echo "Error: nvidia-smi not found but resources monitoring enabled"
	exit 1
    fi
fi

############################
# Utility functions
############################

check_logs_for_errors() {
    local log_dirs=${1:-"logs/step*/pid*"}
    local error_found=0
    local pattern='fatal|fail|exception|traceback'

    for f in $log_dirs/stdout $log_dirs/stderr; do
        [[ -f "$f" ]] || continue

        if grep -qiE "$pattern" "$f"; then
            echo "Error keyword found in: $f"

            grep -inE "$pattern" "$f" | while IFS=: read -r lineno line; do
                keyword=$(grep -ioE "$pattern" <<< "$line" | head -1)
                echo "  Line $lineno [$keyword]: $line"
            done

            error_found=1
        fi
    done

    if [[ $error_found -eq 1 ]]; then
        echo "Failure detected in logs."
        return 1
    fi
}

ensure_patatrack_scripts() {
    if [[ ! -d patatrack-scripts ]]; then
	git clone https://github.com/cms-patatrack/patatrack-scripts --depth 1
    fi
}

get_current_total_gpu_mem() {
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
	| awk '{ total += $1 } END { print total }'
}

get_current_gpus_usage() {
    nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits \
	| paste -sd ','
}

# Helper function to recursively get all child PIDs
get_all_pids() {
    local parent=$1
    echo $parent
    for child in $(pgrep -P $parent 2>/dev/null); do
	get_all_pids $child
    done
}

# Function to get current CPU memory (RSS) in MiB for a process and all its children
get_current_cpu_mem() {
    local pids=$(get_all_pids "$1" | tr '\n' ',' | sed 's/,$//')
    if [[ -n "$pids" ]]; then
	local total_rss_kb=$(ps -o rss= -p "$pids" 2>/dev/null | awk '{sum+=$1} END {print sum}')
	if [[ -n "$total_rss_kb" ]]; then
	    echo $((total_rss_kb / 1024))
	    return
	fi
    fi
    echo "error"
}

############################
# Data handling
############################

fetch_files() {

    # Prefer DAS. It needs a valid X509 proxy, so fall back to listing the files
    # directly under the mounted /eos/cms, deriving the relval path from the
    # dataset name (/<primary>/<release>-<procstring>/<tier>).
    mapfile -t FILES < <(
	dasgoclient -query="file dataset=${DATASET}" --limit=-1 2>/dev/null |
	    sort |
	    head -4
    )

    if [[ ${#FILES[@]} -eq 0 ]]; then
	echo "DAS query returned nothing (expired proxy?), listing /eos/cms directly"
	local primary rest tier release procstring lfndir
	IFS='/' read -r _ primary rest tier <<< "${DATASET}"
	release="${rest%%-*}"
	procstring="${rest#*-}"
	lfndir="/store/relval/${release}/${primary}/${tier}/${procstring}"

	mapfile -t FILES < <(
	    find "/eos/cms${lfndir}" -name '*.root' 2>/dev/null |
		sed 's|^/eos/cms||' |
		sort |
		head -4
	)
    fi

    if [[ ${#FILES[@]} -eq 0 ]]; then
	echo "Error: could not resolve any input file for ${DATASET}"
	return 1
    fi

    for f in "${FILES[@]}"; do

	local mypath
	mypath=$(dirname "$f")

	mkdir -p "${FOLDER_FILES}${mypath}"

	if [[ -e "/eos/cms/$f" && ! -e "${FOLDER_FILES}${f}" ]]; then
	    echo "Copying $f"
	    cp "/eos/cms/$f" "${FOLDER_FILES}${mypath}"
	fi
    done
}

build_input_file_string() {

    LOCALPATH=${FOLDER_FILES}$(dirname ${FILES[0]})

    echo "Local repository: |${LOCALPATH}|"

    ALL_FILES=""

    for f in $(ls -1 ${LOCALPATH}); do
	ALL_FILES+="file:${LOCALPATH}/${f},"
    done

    ALL_FILES="${ALL_FILES%?}"

    echo "Discovered files: $ALL_FILES"
}

############################
# cmsDriver generator
############################

run_cmsdriver() {

    local fragment=$1
    local menu=$2
    local process=$3
    local output_py=$4
    local extra_args=$5

    cmsDriver.py ${fragment} \
		 -s ${menu} \
		 --processName=${process} \
		 --conditions auto:phase2_realistic_T35 \
		 --geometry ExtendedRun4D121 \
		 --era Phase2C22I13M9 \
		 --customise SLHCUpgradeSimulations/Configuration/aging.customise_aging_1000 \
		 --eventcontent FEVTDEBUGHLT \
		 --filein="${ALL_FILES}" \
		 --mc \
		 --nThreads ${THREADS} \
		 --inputCommands 'keep *, drop *_hlt*_*_HLT, drop triggerTriggerFilterObjectWithRefs_l1t*_*_HLT' \
		 -n ${EVENTS} \
		 --no_exec \
		 --output {} \
		 ${extra_args} \
		 --python_filename ${output_py}
}

############################
# Benchmark runner
############################

run_benchmark() {

    local cfg=$1
    local output_json=$2
    local logdir="logs.$(basename ${cfg%.py})"

    if [[ ! -e "$cfg" ]]; then
	echo "Config $cfg not found"
	return
    fi

    ensure_patatrack_scripts
    mkdir -p "$logdir"

    if [[ "$ENABLE_RESOURCES_MONITORING" = true ]]; then

	echo "Running benchmark WITH RESOURCES monitoring"

	local CSV_FILE="${logdir}/gpu_memory.csv"
	local CSV_GPU_FILE="${logdir}/gpu_usage.csv"
	local CSV_CPU_FILE="${logdir}/cpu_memory.csv"
	local TMP_LOG_FILE="${logdir}/benchmark.tmp.log"

	echo "elapsed_seconds,memory_mib" > "$CSV_FILE"
	echo "elapsed_seconds,memory_mib" >"$CSV_CPU_FILE"
	echo "elapsed_seconds,gpu_usage" > "$CSV_GPU_FILE"

	# gpu counters
	local max_mem=0
	local sum_mem=0
	local count=0

	# cpu counters
	local max_mem_cpu=0
	local sum_mem_cpu=0
	local count_cpu=0

	declare -a totals
	declare -a max_usage

	local START_TIME=$(date +%s)

	# Run benchmark in background
	patatrack-scripts/benchmark \
	    -j ${BENCH_JOBS} -t ${BENCH_THREADS} -s ${BENCH_STREAMS} \
	    -e ${EVENTS} \
	    --no-input-benchmark \
	    --slot "${BENCH_SLOT}" \
	    --event-skip 100 \
	    --event-resolution 10 \
	    --debug-logs \
	    -k Phase2Timing_resources.json \
	    -- ${cfg} > "$TMP_LOG_FILE" 2>&1 &

	local PID=$!

	# Ensure cleanup on failure
	trap 'kill $PID 2>/dev/null || true' EXIT

	# Live output
	tail -f --pid=$PID "$TMP_LOG_FILE" &
	local TAIL_PID=$!

	while kill -0 $PID 2>/dev/null; do

	    # Memory
	    mem=$(get_current_total_gpu_mem)
	    now=$(date +%s)
	    elapsed=$((now - START_TIME))

	    if [[ "$mem" =~ ^[0-9]+$ ]]; then
		echo "$elapsed,$mem" >> "$CSV_FILE"
		((mem > max_mem)) && max_mem=$mem
		sum_mem=$((sum_mem + mem))
		count=$((count + 1))
	    fi

	    # CPU Memory Monitor
	    cpu_mem=$(get_current_cpu_mem $PID)
	    if [[ "$cpu_mem" =~ ^[0-9]+$ ]]; then
		echo "$elapsed,$cpu_mem" >>"$CSV_CPU_FILE"
		if ((cpu_mem > max_mem_cpu)); then max_mem_cpu=$cpu_mem; fi
		sum_mem_cpu=$((sum_mem_cpu + cpu_mem))
		count_cpu=$((count_cpu + 1))
	    fi

	    # GPU usage
	    usage=$(get_current_gpus_usage)
	    if [[ "$usage" =~ ^[0-9,]+$ ]]; then
		echo "$elapsed,$usage" >> "$CSV_GPU_FILE"

		IFS=',' read -ra vals <<< "$usage"
		for i in "${!vals[@]}"; do
		    totals[$i]=$((${totals[$i]:-0} + vals[$i]))
		    ((vals[$i] > ${max_usage[$i]:-0})) && max_usage[$i]=${vals[$i]}
		done
	    fi

	    sleep $MONITOR_INTERVAL
	done

	wait $PID

	#tail should already exit due to --pid=$PID
	wait $TAIL_PID 2>/dev/null || true

	mv "$TMP_LOG_FILE" "${logdir}/output.log"

	# Compute GPU memory mean
	if ((count > 0)); then
	    mean_mem=$((sum_mem / count))
	else
	    mean_mem=0
	fi

	# Compute CPU memory mean
	if ((count_cpu > 0)); then
	    mean_mem_cpu=$((sum_mem_cpu / count_cpu))
	else
	    mean_mem_cpu=0
	fi

	{
	    echo ""
	    echo "----- HARDWARE USAGE SUMMARY -----"
	    echo "Peak CPU memory: ${max_mem_cpu} MiB"
	    echo "Mean CPU memory: ${mean_mem_cpu} MiB"
	    echo ""
	    echo "Peak GPU memory: ${max_mem} MiB"
	    echo "Mean GPU memory: ${mean_mem} MiB"
	    echo ""
	    echo "Per-GPU usage:"
	    for i in "${!totals[@]}"; do
		avg=$((totals[$i] / count))
		echo "GPU $i: avg=${avg}% max=${max_usage[$i]}%"
	    done
	    echo "----------------------------------"
	} | tee -a "${logdir}/output.log"

    else

	echo "Running benchmark WITHOUT RESOURCES monitoring"

	patatrack-scripts/benchmark \
	    -j ${BENCH_JOBS} -t ${BENCH_THREADS} -s ${BENCH_STREAMS} \
	    -e ${EVENTS} \
	    --no-input-benchmark \
	    --slot "${BENCH_SLOT}" \
	    --event-skip 100 \
	    --event-resolution 10 \
	    --debug-logs \
	    -k Phase2Timing_resources.json \
	    -- ${cfg} | tee "${logdir}/output.log"
    fi

    check_logs_for_errors || exit 1

    mergeResourcesJson.py logs/step*/pid*/Phase2Timing_resources.json > "${output_json}"
}

############################
# Workflows
############################

run_phase2_gpu() {

    run_cmsdriver \
	"Phase2" \
	"L1P2GT,HLT:75e33_timing" \
	"HLTX" \
	"Phase2_L1P2GT_HLT.py" \
	""

    run_benchmark \
	"Phase2_L1P2GT_HLT.py" \
	"Phase2Timing_resources.json"

    if [[ -e "$(dirname $0)/augmentResources.py" ]]; then
	python3 $(dirname $0)/augmentResources.py
    fi
}

run_phase2_cpu() {

    run_cmsdriver \
	"Phase2" \
	"L1P2GT,HLT:75e33_timing" \
	"HLTX" \
	"Phase2_L1P2GT_HLT_OnCPU.py" \
	"--accelerators cpu"

    run_benchmark \
	"Phase2_L1P2GT_HLT_OnCPU.py" \
	"Phase2Timing_resources_OnCPU.json"
}

run_ngt_scouting() {

    run_cmsdriver \
	"NGTScouting" \
	"L1P2GT,HLT:NGTScouting" \
	"NLTX" \
	"NGTScouting_L1P2GT_HLT.py" \
	"--procModifiers alpaka,ngtScouting"

    run_benchmark \
	"NGTScouting_L1P2GT_HLT.py" \
	"Phase2Timing_resources_NGT.json"
}

############################
# Main
############################

main() {
    fetch_files
    build_input_file_string

    run_phase2_gpu
    run_phase2_cpu
    run_ngt_scouting
}

main "$@"
