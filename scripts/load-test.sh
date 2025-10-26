#!/bin/bash
set -e

HOST="localhost"
PORT=8001
REPORT_DIR="reports"
USERS=50
SPAWN_RATE=10
DURATION="30s"

usage() {
    echo "Usage: $0 [-H host] [-P port] [-u users] [-r spawn_rate] [-d duration]"
    echo "  -H  Host (default: $HOST)"
    echo "  -P  Port (default: $PORT)"
    echo "  -u  Number of users (default: $USERS)"
    echo "  -r  Spawn rate (default: $SPAWN_RATE)"
    echo "  -d  Test duration (default: $DURATION)"
    exit 1
}

while getopts "H:P:u:r:d:" opt; do
    case "$opt" in
        H) HOST="$OPTARG" ;;
        P) PORT="$OPTARG" ;;
        u) USERS="$OPTARG" ;;
        r) SPAWN_RATE="$OPTARG" ;;
        d) DURATION="$OPTARG" ;;
        *) usage ;;
    esac
done

REPORT_FILENAME="report-$(date +"%Y-%m-%d-%H-%M").html"

mkdir -p "$REPORT_DIR"

FULL_HOST="http://$HOST:$PORT"

echo "Running load test for $DURATION with $USERS users at $SPAWN_RATE spawn rate against $FULL_HOST..."

uv run locust -f src/benchmarks/locustfile.py \
    --host="$FULL_HOST" \
    --users="$USERS" \
    --spawn-rate="$SPAWN_RATE" \
    --run-time="$DURATION" \
    --headless \
    --html="$REPORT_DIR/$REPORT_FILENAME" \
    --loglevel=ERROR \
    --only-summary

# symlink for accessing latest results
ln -sf "$REPORT_FILENAME" "$REPORT_DIR/latest.html"

echo "Load test complete. Results stored in $REPORT_DIR/$REPORT_FILENAME"
