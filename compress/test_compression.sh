#!/bin/bash
# Compression test script for RomLib

echo "=== RomLib Compression Test ==="
echo ""

# Test with 30MB file
TEST_FILE="test_data/raw_games/test-game-001/main.rom"
if [ ! -f "$TEST_FILE" ]; then
    echo "Creating 30MB test file..."
    mkdir -p test_data/raw_games/test-game-001
    dd if=/dev/urandom of="$TEST_FILE" bs=1M count=30 2>/dev/null
fi

ORIGINAL_SIZE=$(stat -f%z "$TEST_FILE" 2>/dev/null || stat -c%s "$TEST_FILE" 2>/dev/null)
ORIGINAL_SIZE_MB=$(echo "scale=2; $ORIGINAL_SIZE / 1024 / 1024" | bc)

echo "Original file size: ${ORIGINAL_SIZE_MB} MB ($(numfmt --to=iec-i --suffix=B $ORIGINAL_SIZE))"
echo ""

# Test zstd compression directly to estimate
echo "Testing zstd compression (level 9)..."
COMPRESSED=$(zstd -c -9 "$TEST_FILE" 2>/dev/null | wc -c)
COMPRESSED_SIZE_MB=$(echo "scale=2; $COMPRESSED / 1024 / 1024" | bc)
COMPRESSION_RATIO=$(echo "scale=2; $ORIGINAL_SIZE / $COMPRESSED" | bc)

echo "Zstd compressed size: ${COMPRESSED_SIZE_MB} MB ($(numfmt --to=iec-i --suffix=B $COMPRESSED))"
echo "Compression ratio: ${COMPRESSION_RATIO}x"
echo ""

# Estimate chunking overhead
# With 32KB average chunks, a 30MB file = ~937 chunks
# Each chunk header = 44 bytes
CHUNKS=$(echo "scale=0; $ORIGINAL_SIZE / 32768" | bc)
HEADER_OVERHEAD=$(echo "$CHUNKS * 44" | bc)
HEADER_OVERHEAD_KB=$(echo "scale=2; $HEADER_OVERHEAD / 1024" | bc)

echo "Estimated chunking:"
echo "  Chunks: ~$CHUNKS (32KB average)"
echo "  Header overhead: ${HEADER_OVERHEAD_KB} KB"
echo ""

# Estimate total pack size
TOTAL_PACK_SIZE=$(echo "$COMPRESSED + $HEADER_OVERHEAD" | bc)
TOTAL_PACK_SIZE_MB=$(echo "scale=2; $TOTAL_PACK_SIZE / 1024 / 1024" | bc)

echo "Estimated total pack size:"
echo "  Compressed data: ${COMPRESSED_SIZE_MB} MB"
echo "  + Headers: ${HEADER_OVERHEAD_KB} KB"
echo "  = Total: ${TOTAL_PACK_SIZE_MB} MB ($(numfmt --to=iec-i --suffix=B $TOTAL_PACK_SIZE))"
echo ""

# Estimate manifest size
# Each chunk hash = 32 bytes
# For 937 chunks = ~30KB of hashes
# Plus metadata = ~1-2 KB
MANIFEST_HASHES=$(echo "$CHUNKS * 32" | bc)
MANIFEST_METADATA=2048
MANIFEST_TOTAL=$(echo "$MANIFEST_HASHES + $MANIFEST_METADATA" | bc)
MANIFEST_KB=$(echo "scale=2; $MANIFEST_TOTAL / 1024" | bc)

echo "Estimated manifest size:"
echo "  Chunk hashes: $(echo "scale=2; $MANIFEST_HASHES / 1024" | bc) KB"
echo "  Metadata: $(echo "scale=2; $MANIFEST_METADATA / 1024" | bc) KB"
echo "  Total: ${MANIFEST_KB} KB"
echo ""
echo "Summary for 30MB folder:"
echo "  Original: ${ORIGINAL_SIZE_MB} MB"
echo "  Pack (.pkg): ~${TOTAL_PACK_SIZE_MB} MB (with deduplication, could be smaller)"
echo "  Manifest: ~${MANIFEST_KB} KB"
echo ""
echo "If this folder is already in packs (deduplicated), the manifest alone"
echo "is enough to reconstruct it - just ${MANIFEST_KB} KB instead of ${ORIGINAL_SIZE_MB} MB!"
