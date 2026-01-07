#!/bin/bash
# Transfer script: aip1 -> aip2
# Run this from a LOGIN NODE (not compute node)

SOURCE="/lus/lfs1aip1/home/a5l/shuqing.a5l/"
DEST="a5l.aip2.isambard:/lus/lfs1aip2/home/a5l/shuqing.a5l/"

# Log file
LOGFILE="/lus/lfs1aip1/home/a5l/shuqing.a5l/transfer_log_$(date +%Y%m%d_%H%M%S).log"

echo "=============================================="
echo "Transfer: aip1 -> aip2"
echo "Source: $SOURCE"
echo "Dest:   $DEST"
echo "Log:    $LOGFILE"
echo "=============================================="

# Check if we're on a login node (can resolve external hosts)
if ! ssh -o BatchMode=yes -o ConnectTimeout=5 a5l.aip2.isambard "echo 'Connection OK'" 2>/dev/null; then
    echo "ERROR: Cannot connect to aip2. Are you on a login node?"
    echo "Run 'exit' to get back to login node first."
    exit 1
fi

echo ""
echo "Choose transfer mode:"
echo "  1) Full transfer (everything)"
echo "  2) Exclude cache/temp files (faster, recommended)"
echo "  3) Dry run (show what would be transferred)"
echo ""
read -p "Enter choice [1-3]: " choice

case $choice in
    1)
        echo "Starting FULL transfer..."
        rsync -avz --progress \
            "$SOURCE" "$DEST" 2>&1 | tee "$LOGFILE"
        ;;
    2)
        echo "Starting transfer (excluding cache/temp)..."
        rsync -avz --progress \
            --exclude '.cache/' \
            --exclude '.triton/' \
            --exclude '.triton_aarch64/' \
            --exclude '.triton_build_nano/' \
            --exclude '.pip-tmp/' \
            --exclude '.tmp-build/' \
            --exclude '.nv/' \
            --exclude '__pycache__/' \
            --exclude '*.pyc' \
            --exclude 'slurm-*.out' \
            --exclude '.vscode-server/' \
            --exclude '.cursor-server/' \
            --exclude 'outputs/' \
            --exclude 'logs/' \
            "$SOURCE" "$DEST" 2>&1 | tee "$LOGFILE"
        ;;
    3)
        echo "DRY RUN (no files transferred)..."
        rsync -avz --dry-run --progress \
            --exclude '.cache/' \
            --exclude '.triton/' \
            --exclude '.triton_aarch64/' \
            --exclude '.triton_build_nano/' \
            --exclude '.pip-tmp/' \
            --exclude '.tmp-build/' \
            --exclude '.nv/' \
            --exclude '__pycache__/' \
            --exclude '*.pyc' \
            --exclude 'slurm-*.out' \
            --exclude '.vscode-server/' \
            --exclude '.cursor-server/' \
            "$SOURCE" "$DEST" 2>&1 | tee "$LOGFILE"
        ;;
    *)
        echo "Invalid choice. Exiting."
        exit 1
        ;;
esac

echo ""
echo "=============================================="
echo "Transfer complete!"
echo "Log saved to: $LOGFILE"
echo "=============================================="
