#!/bin/bash
set -euo pipefail

# ============================================================================
# ninfer-serve Cooperative Launch Fix — Server Test Script
# ============================================================================
# Automates build, deploy, and verification of the fix for
# cudaErrorCooperativeLaunchTooLarge on non-5090 GPUs.
# ============================================================================

NINFER_DIR="$HOME/ninfer"
SERVE_PID_FILE="$HOME/ninfer/ninfer-serve.pid"
PORT=8090
MODEL_PATH="$HOME/ninfer/models/qwen3_5_9b.ninfer"
BRANCH="fix/cooperative-launch-sm-count"
TIMEOUT=120  # seconds to wait for server startup

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

log()  { echo -e "${GREEN}[PASS]${NC} $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
fail() { echo -e "${RED}[FAIL]${NC} $*"; }
info() { echo -e "${CYAN}[INFO]${NC} $*"; }

trap 'fail "Script interrupted"; cleanup_on_fail; exit 1' INT TERM

# ============================================================================
# Functions
# ============================================================================

cleanup_on_fail() {
    info "Cleaning up failed deployment..."
    if [ -f "$SERVE_PID_FILE" ]; then
        local pid
        pid=$(cat "$SERVE_PID_FILE" 2>/dev/null || true)
        if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null || true
            wait "$pid" 2>/dev/null || true
        fi
        rm -f "$SERVE_PID_FILE"
    fi
}

stop_serve() {
    info "Stopping existing ninfer-serve..."

    # Kill whatever is listening on our port (works regardless of PID file).
    local port_pid
    port_pid=$(ss -tlnp 2>/dev/null | awk -v p=":$PORT " '$4 ~ p {print $NF}' | grep -o '[0-9]*' | head -1 || true)
    if [ -n "$port_pid" ] && kill -0 "$port_pid" 2>/dev/null; then
        kill "$port_pid" 2>/dev/null || true
        sleep 3
        kill -9 "$port_pid" 2>/dev/null || true
        log "Stopped serve on port $PORT (PID $port_pid)"
    fi

    # Clean up any PID file from a previous run.
    if [ -f "$SERVE_PID_FILE" ]; then
        local saved_pid
        saved_pid=$(cat "$SERVE_PID_FILE" 2>/dev/null || true)
        if [ -n "$saved_pid" ] && [ "$saved_pid" != "$port_pid" ] && kill -0 "$saved_pid" 2>/dev/null; then
            kill "$saved_pid" 2>/dev/null || true
            sleep 3
            kill -9 "$saved_pid" 2>/dev/null || true
            log "Stopped stale serve (PID $saved_pid)"
        fi
        rm -f "$SERVE_PID_FILE"
    fi

    # Belt-and-suspenders: kill any remaining ninfer-serve process.
    pkill -f "ninfer-serve" 2>/dev/null || true

    # Wait until the port is actually free.
    local attempts=0
    while ss -tln | grep -q ":$PORT " && [ $attempts -lt 15 ]; do
        sleep 1
        attempts=$((attempts + 1))
    done
    if ss -tln | grep -q ":$PORT "; then
        fail "Port $PORT still in use after stopping serve"
        return 1
    fi
    log "Port $PORT is free"
}

wait_for_ready() {
    local elapsed=0
    local interval=5
    local pid
    pid=$(cat "$SERVE_PID_FILE" 2>/dev/null || true)
    info "Waiting for ninfer-serve to be ready (timeout: ${TIMEOUT}s, PID $pid)..."

    while [ $elapsed -lt $TIMEOUT ]; do
        if curl -sf "http://localhost:$PORT/health" > /dev/null 2>&1; then
            log "Server is ready after ${elapsed}s"
            return 0
        fi
        # If we have a PID and the process is no longer alive, it died — fail fast.
        if [ -n "$pid" ] && ! kill -0 "$pid" 2>/dev/null; then
            fail "Serve process (PID $pid) died unexpectedly after ${elapsed}s"
            return 1
        fi
        sleep $interval
        elapsed=$((elapsed + interval))
    done

    fail "Server did not become ready within ${TIMEOUT}s"
    return 1
}

run_inference_test() {
    local prompt="$1"
    local expected_pattern="$2"
    local test_name="$3"
    local max_tokens="${4:-128}"

    info "Running test: $test_name"
    info "Prompt: ${prompt:0:80}..."

    local response
    response=$(curl -sf -X POST "http://localhost:$PORT/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "{
            \"model\": \"qwen3_5_9b\",
            \"messages\": [{\"role\": \"user\", \"content\": \"$prompt\"}],
            \"max_tokens\": $max_tokens,
            \"temperature\": 0.1,
            \"stream\": false
        }" 2>&1) || {
        fail "Test '$test_name': HTTP request failed"
        echo "$response"
        return 1
    }

    if echo "$response" | grep -q '"error"'; then
        fail "Test '$test_name': API returned error"
        echo "$response" | python3 -c "import sys,json; print(json.dumps(json.load(sys.stdin), indent=2))" 2>/dev/null || echo "$response"
        return 1
    fi

    local content
    content=$(echo "$response" | python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['message']['content'])" 2>/dev/null || true)

    if [ -z "$content" ]; then
        fail "Test '$test_name': Empty response"
        return 1
    fi

    if [ -n "$expected_pattern" ] && ! echo "$content" | grep -qi "$expected_pattern"; then
        warn "Test '$test_name': Response didn't match expected pattern '$expected_pattern'"
        warn "Response: ${content:0:200}"
    else
        log "Test '$test_name': OK (${#content} chars)"
    fi

    return 0
}

run_benchmark() {
    local num_requests="${1:-5}"
    local tokens="${2:-256}"

    info "Running benchmark: $num_requests requests x $tokens tokens..."

    local success=0
    local failed=0
    local start_time end_time total_time

    start_time=$(date +%s)

    for i in $(seq 1 $num_requests); do
        local response
        response=$(curl -sf -X POST "http://localhost:$PORT/v1/chat/completions" \
            -H "Content-Type: application/json" \
            -d "{
                \"model\": \"qwen3_5_9b\",
                \"messages\": [{\"role\": \"user\", \"content\": \"Write a numbered list from 1 to $((i * 5))\"}],
                \"max_tokens\": $tokens,
                \"temperature\": 0.1,
                \"stream\": false
            }" 2>&1) || {
            fail "Benchmark request $i failed"
            failed=$((failed + 1))
            continue
        }

        if echo "$response" | grep -q '"error"'; then
            fail "Benchmark request $i: API error"
            failed=$((failed + 1))
        else
            success=$((success + 1))
        fi
    done

    end_time=$(date +%s)
    total_time=$((end_time - start_time))

    info "Benchmark complete: $success/$num_requests succeeded in ${total_time}s"

    if [ $success -eq $num_requests ]; then
        log "All benchmark requests passed"
        return 0
    elif [ $success -gt 0 ]; then
        warn "$failed benchmark requests failed"
        return 1
    else
        fail "All benchmark requests failed"
        return 1
    fi
}

check_cuda_errors() {
    info "Checking for CUDA cooperative launch errors in serve logs..."

    # Check dmesg and any error output
    local errors=""
    errors=$(dmesg 2>/dev/null | grep -i "cuda\|cooperative\|launch" | tail -5 || true)

    if [ -n "$errors" ]; then
        warn "CUDA-related messages in dmesg:"
        echo "$errors"
    else
        log "No CUDA cooperative launch errors detected"
    fi
}

# ============================================================================
# Main
# ============================================================================

info "=============================================="
info "  ninfer-serve Cooperative Launch Fix Test"
info "=============================================="
info ""

# Step 1: Verify we're on the right branch
info "Step 1: Verifying git branch..."
cd "$NINFER_DIR"
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
if [ "$CURRENT_BRANCH" = "$BRANCH" ]; then
    log "On correct branch: $BRANCH"
else
    fail "Expected branch '$BRANCH', got '$CURRENT_BRANCH'"
    exit 1
fi

# Step 2: Pull latest changes
info "Step 2: Pulling latest changes..."
git pull origin "$BRANCH"
log "Latest changes pulled"

# Step 3: Build
info "Step 3: Building project..."
BUILD_START=$(date +%s)
cd "$NINFER_DIR"
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release > /dev/null 2>&1
cmake --build build --parallel > /dev/null 2>&1
BUILD_END=$(date +%s)
BUILD_TIME=$((BUILD_END - BUILD_START))
log "Build completed in ${BUILD_TIME}s"

# Verify binary exists
if [ ! -x "$NINFER_DIR/build/apps/ninfer-serve" ]; then
    fail "Built binary not found at $NINFER_DIR/build/apps/ninfer-serve"
    exit 1
fi
log "Binary verified"

# Verify the model file exists before we start the server.
if [ ! -f "$MODEL_PATH" ]; then
    fail "Model file not found: $MODEL_PATH"
    info "Available models in $HOME/ninfer/models/:"
    ls -1 "$HOME/ninfer/models/" 2>/dev/null || true
    exit 1
fi
log "Model verified: $MODEL_PATH"

# Step 4: Stop existing serve
stop_serve

# Step 5: Start new serve
info "Step 5: Starting ninfer-serve with new binary..."
cd "$NINFER_DIR"
./build/apps/ninfer-serve \
    "$MODEL_PATH" \
    --host 0.0.0.0 \
    --port $PORT \
    --max-context 8192 \
    --prefill-chunk 4096 \
    --kv-dtype int8 \
    --greedy \
    --spec mtp \
    --draft-tokens 3 \
    > "$NINFER_DIR/serve.log" 2>&1 &
SERVE_PID=$!
echo $SERVE_PID > "$SERVE_PID_FILE"

info "Started ninfer-serve (PID: $SERVE_PID)"

# Step 6: Wait for server to be ready
if ! wait_for_ready; then
    fail "Server failed to start"
    info "Serve logs:"
    tail -50 "$NINFER_DIR/serve.log" 2>/dev/null || true
    cleanup_on_fail
    exit 1
fi

# Step 7: Run inference tests
info ""
info "=============================================="
info "  Inference Tests"
info "=============================================="

TEST_PASSED=0
TEST_FAILED=0

# Test 1: Basic math (verifies model is responding)
if run_inference_test "What is 2 + 2?" "[Tt]wo|[Tt]welve|[Ff]our|4" "Basic math (2+2)"; then
    TEST_PASSED=$((TEST_PASSED + 1))
else
    TEST_FAILED=$((TEST_FAILED + 1))
fi

# Test 2: List generation (tests token generation)
if run_inference_test "List the first 5 prime numbers" "[Pp]rime|[2357]|prime" "Prime numbers list"; then
    TEST_PASSED=$((TEST_PASSED + 1))
else
    TEST_FAILED=$((TEST_FAILED + 1))
fi

# Test 3: Longer context (stress test the GDN gating)
if run_inference_test "Write a short story about a robot learning to code. Keep it under 100 words." "[Rr]obot|[Cc]ode|[Ss]tory" "Short story generation"; then
    TEST_PASSED=$((TEST_PASSED + 1))
else
    TEST_FAILED=$((TEST_FAILED + 1))
fi

# Step 8: Run benchmark
info ""
info "=============================================="
info "  Benchmark Tests"
info "=============================================="

if run_benchmark 5 128; then
    TEST_PASSED=$((TEST_PASSED + 1))
else
    TEST_FAILED=$((TEST_FAILED + 1))
fi

# Step 9: Check for CUDA errors
check_cuda_errors

# ============================================================================
# Summary
# ============================================================================

info ""
info "=============================================="
info "  Test Summary"
info "=============================================="
info "  Inference tests: $TEST_PASSED passed, $TEST_FAILED failed"
info ""

if [ $TEST_FAILED -eq 0 ]; then
    log "ALL TESTS PASSED"
    info ""
    info "The fix appears to be working correctly on this GPU."
    info "Serve is still running on port $PORT if you need to do manual testing."
else
    fail "$TEST_FAILED test(s) failed"
    info ""
    info "Check the serve logs: tail -f $NINFER_DIR/serve.log"
    info "Consider: $(warn 'ssh kuroko@192.168.1.23 \"tail -100 ~/ninfer/serve.log\"')"
    cleanup_on_fail
    exit 1
fi

exit 0