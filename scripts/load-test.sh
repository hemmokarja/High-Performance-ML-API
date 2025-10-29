#!/bin/bash
set -e

HOST="localhost"
PORT=8000  # gateway port
REPORT_DIR="reports"
USERS=50
SPAWN_RATE=10
DURATION="30s"
MAX_HEALTH_RETRIES=20
HEALTH_RETRY_DELAY=5
LOCUSTFILE="src/benchmarks/locustfile_gateway.py"  # default locustfile path

usage() {
    echo "Usage: $0 [-H host] [-P port] [-u users] [-r spawn_rate] [-d duration] [-f locustfile]"
    echo "  -H  Host (default: $HOST)"
    echo "  -P  Port (default: $PORT)"
    echo "  -u  Number of users (default: $USERS)"
    echo "  -r  Spawn rate (default: $SPAWN_RATE)"
    echo "  -d  Test duration (default: $DURATION)"
    echo "  -f  Locustfile path (default: $LOCUSTFILE)"
    exit 1
}

while getopts "H:P:u:r:d:f:" opt; do
    case "$opt" in
        H) HOST="$OPTARG" ;;
        P) PORT="$OPTARG" ;;
        u) USERS="$OPTARG" ;;
        r) SPAWN_RATE="$OPTARG" ;;
        d) DURATION="$OPTARG" ;;
        f) LOCUSTFILE="$OPTARG" ;;
        *) usage ;;
    esac
done

if [ ! -f "$LOCUSTFILE" ]; then
    echo "Error: Locustfile not found at $LOCUSTFILE"
    exit 1
fi

FULL_HOST="http://$HOST:$PORT"
HEALTH_URL="$FULL_HOST/health"
REPORT_FILENAME="report-$(date +"%Y-%m-%d-%H-%M").html"

# Health check with retries
echo "Checking health..."
for i in $(seq 1 $MAX_HEALTH_RETRIES); do
    if curl -sf "$HEALTH_URL" > /dev/null 2>&1; then
        echo "Service ready!"
        break
    else
        if [ $i -eq $MAX_HEALTH_RETRIES ]; then
            echo "Health check failed - is the service running?"
            exit 1
        fi
        sleep $HEALTH_RETRY_DELAY
    fi
done

mkdir -p "$REPORT_DIR"

echo "Running load test for $DURATION with $USERS users at $SPAWN_RATE spawn rate against $FULL_HOST using locustfile: $LOCUSTFILE..."

uv run locust -f "$LOCUSTFILE" \
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
