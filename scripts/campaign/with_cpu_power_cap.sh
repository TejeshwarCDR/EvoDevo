#!/usr/bin/env bash
# Run one supervised child under reversible, host-wide Intel CPU power controls.
#
# This script deliberately refuses to start without root and restores every
# value it changes on normal exit, error, or INT/TERM.  It is for controlled
# benchmark runs only: do not wrap the scientific campaign with it.
#
# Example (an operator supplies the sudo credential interactively):
#   sudo scripts/campaign/with_cpu_power_cap.sh --p-max-mhz 4200 --p-cpus 0-15 \
#     -- .venv/bin/python scripts/campaign/benchmark_thermal_affinity.py \
#     --profiles p8e12 --genomes 960 --abort-temp-c 90 --control-label pmax_4200mhz

set -euo pipefail

PSTATE=/sys/devices/system/cpu/intel_pstate
RAPL=/sys/class/powercap/intel-rapl:0
MAX_PERF_PCT=""
NO_TURBO=""
TURBO=""
PL1_W=""
PL2_W=""
P_MAX_MHZ=""
P_CPUS=""
DRY_RUN=0
RUN_AS_USER="${SUDO_USER:-}"
declare -a COMMAND=()
declare -a ORIGINAL_PATHS=()
declare -a ORIGINAL_VALUES=()

usage() {
    sed -n '2,16p' "$0" | sed 's/^# \{0,1\}//'
    cat <<'EOF'

Options:
  --max-perf-pct N  Set intel_pstate's global maximum performance percentage (1..100).
  --no-turbo        Disable Intel turbo while the child command runs.
  --turbo           Require Intel turbo enabled while the child command runs.
  --p-max-mhz N     Cap the selected P-core CPU policies at N MHz.
  --p-cpus LIST     CPUs whose policies are capped; required with --p-max-mhz.
                    LIST accepts comma-separated IDs and inclusive ranges (e.g. 0-15).
  --pl1-w W         Set package long-term RAPL limit in watts (positive integer).
  --pl2-w W         Set package short-term RAPL limit in watts (positive integer).
  --run-as-user U   Run the child as U (default: the sudo invoker).
  --run-as-root     Run the benchmark child as root (normally unnecessary).
  --dry-run         Validate and print the intended mutation; do not write or run.
  --                Required separator before the benchmark command.
EOF
}

die() { echo "error: $*" >&2; exit 2; }
is_positive_int() { [[ "$1" =~ ^[1-9][0-9]*$ ]]; }

expand_cpu_list() {
    local part first last cpu
    IFS=',' read -r -a parts <<<"$1"
    for part in "${parts[@]}"; do
        if [[ "$part" =~ ^([0-9]+)-([0-9]+)$ ]]; then
            first=${BASH_REMATCH[1]}; last=${BASH_REMATCH[2]}
            ((first <= last)) || die "invalid descending CPU range: $part"
            for ((cpu=first; cpu<=last; ++cpu)); do printf '%s\n' "$cpu"; done
        elif [[ "$part" =~ ^[0-9]+$ ]]; then
            printf '%s\n' "$part"
        else
            die "invalid CPU list item: $part"
        fi
    done
}

while (($#)); do
    case "$1" in
        --max-perf-pct) (($# >= 2)) || die "--max-perf-pct needs a value"; MAX_PERF_PCT=$2; shift 2 ;;
        --no-turbo) NO_TURBO=1; shift ;;
        --turbo) TURBO=1; shift ;;
        --p-max-mhz) (($# >= 2)) || die "--p-max-mhz needs a value"; P_MAX_MHZ=$2; shift 2 ;;
        --p-cpus) (($# >= 2)) || die "--p-cpus needs a value"; P_CPUS=$2; shift 2 ;;
        --pl1-w) (($# >= 2)) || die "--pl1-w needs a value"; PL1_W=$2; shift 2 ;;
        --pl2-w) (($# >= 2)) || die "--pl2-w needs a value"; PL2_W=$2; shift 2 ;;
        --run-as-user) (($# >= 2)) || die "--run-as-user needs a user"; RUN_AS_USER=$2; shift 2 ;;
        --run-as-root) RUN_AS_USER=""; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        --help|-h) usage; exit 0 ;;
        --) shift; COMMAND=("$@"); break ;;
        *) die "unknown option: $1" ;;
    esac
done

[[ -n "$MAX_PERF_PCT$NO_TURBO$TURBO$PL1_W$PL2_W$P_MAX_MHZ" ]] || die "select at least one control"
[[ -z "$NO_TURBO" || -z "$TURBO" ]] || die "--turbo and --no-turbo are mutually exclusive"
[[ -n "$MAX_PERF_PCT" ]] && { is_positive_int "$MAX_PERF_PCT" && ((MAX_PERF_PCT <= 100)); } || [[ -z "$MAX_PERF_PCT" ]] || die "--max-perf-pct must be 1..100"
[[ -n "$PL1_W" ]] && is_positive_int "$PL1_W" || [[ -z "$PL1_W" ]] || die "--pl1-w must be a positive integer"
[[ -n "$PL2_W" ]] && is_positive_int "$PL2_W" || [[ -z "$PL2_W" ]] || die "--pl2-w must be a positive integer"
[[ -n "$P_MAX_MHZ" ]] && is_positive_int "$P_MAX_MHZ" || [[ -z "$P_MAX_MHZ" ]] || die "--p-max-mhz must be a positive integer"
[[ -z "$P_MAX_MHZ" && -z "$P_CPUS" ]] || [[ -n "$P_MAX_MHZ" && -n "$P_CPUS" ]] || die "--p-max-mhz and --p-cpus must be used together"
((DRY_RUN)) || ((${#COMMAND[@]})) || die "a command after -- is required"

plan=()
[[ -n "$MAX_PERF_PCT" ]] && plan+=("$PSTATE/max_perf_pct=$MAX_PERF_PCT")
[[ -n "$NO_TURBO" ]] && plan+=("$PSTATE/no_turbo=1")
[[ -n "$TURBO" ]] && plan+=("$PSTATE/no_turbo=0")
[[ -n "$PL1_W" ]] && plan+=("$RAPL/constraint_0_power_limit_uw=$((PL1_W * 1000000))")
[[ -n "$PL2_W" ]] && plan+=("$RAPL/constraint_1_power_limit_uw=$((PL2_W * 1000000))")
if [[ -n "$P_MAX_MHZ" ]]; then
    while IFS= read -r cpu; do
        path="/sys/devices/system/cpu/cpufreq/policy${cpu}/scaling_max_freq"
        [[ -e "$path" ]] || die "CPU $cpu has no cpufreq policy at $path"
        plan+=("$path=$((P_MAX_MHZ * 1000))")
    done < <(expand_cpu_list "$P_CPUS" | sort -nu)
fi

# Fail before a privileged mutation when a requested kernel interface is not
# exposed.  This also makes --dry-run an honest capability check.
for assignment in "${plan[@]}"; do
    path=${assignment%%=*}
    [[ -e "$path" ]] || die "requested control is unavailable: $path"
done

if ((DRY_RUN)); then
    printf 'would set %s\n' "${plan[@]}"
    ((${#COMMAND[@]})) && printf 'would run:' && printf ' %q' "${COMMAND[@]}" && printf '\n'
    exit 0
fi

((EUID == 0)) || die "root is required; invoke this controlled benchmark with sudo"
[[ -z "$RUN_AS_USER" ]] || id "$RUN_AS_USER" >/dev/null 2>&1 || die "unknown benchmark user: $RUN_AS_USER"
for assignment in "${plan[@]}"; do
    path=${assignment%%=*}
    [[ -w "$path" ]] || die "control is not writable: $path"
done

restore() {
    local i
    for ((i=${#ORIGINAL_PATHS[@]} - 1; i >= 0; --i)); do
        printf '%s' "${ORIGINAL_VALUES[i]}" > "${ORIGINAL_PATHS[i]}" || \
            echo "WARNING: could not restore ${ORIGINAL_PATHS[i]}" >&2
    done
}
on_signal() { trap - INT TERM HUP; restore; exit 128; }
trap restore EXIT
trap on_signal INT TERM HUP

for assignment in "${plan[@]}"; do
    path=${assignment%%=*}
    value=${assignment#*=}
    ORIGINAL_PATHS+=("$path")
    ORIGINAL_VALUES+=("$(<"$path")")
    printf '%s' "$value" > "$path"
done

# sysfs writes can be accepted but clamped or ignored by firmware.  A
# production launcher must fail closed instead of merely trusting a successful
# write syscall.
for assignment in "${plan[@]}"; do
    path=${assignment%%=*}
    value=${assignment#*=}
    actual="$(<"$path")"
    [[ "$actual" == "$value" ]] || die "control verification failed: $path expected $value, observed $actual"
done

printf 'applied reversible controls:'; printf ' %s' "${plan[@]}"; printf '\n'
CHILD_PID=""
on_signal() {
    trap - INT TERM HUP
    if [[ -n "$CHILD_PID" ]] && kill -0 "$CHILD_PID" 2>/dev/null; then
        kill -INT "$CHILD_PID" 2>/dev/null || true
        wait "$CHILD_PID" || true
    fi
    trap - EXIT
    restore
    exit 128
}
trap on_signal INT TERM HUP
if [[ -n "$RUN_AS_USER" ]]; then
    printf 'running child as %s\n' "$RUN_AS_USER"
    runuser -u "$RUN_AS_USER" -- "${COMMAND[@]}" &
else
    "${COMMAND[@]}" &
fi
CHILD_PID=$!
wait "$CHILD_PID"
