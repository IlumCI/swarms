#!/usr/bin/env python3
"""Compression test for RomLib - demonstrates compression on a 30MB folder"""

import os
import subprocess
import sys

def format_size(size_bytes):
    """Format bytes to human readable"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"

print("=== RomLib Compression Test ===\n")

# Test with 30MB file
test_file = "test_data/raw_games/test-game-001/main.rom"
if not os.path.exists(test_file):
    print("Creating 30MB test file...")
    os.makedirs(os.path.dirname(test_file), exist_ok=True)
    subprocess.run(["dd", "if=/dev/urandom", f"of={test_file}", "bs=1M", "count=30"], 
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

original_size = os.path.getsize(test_file)
original_size_mb = original_size / (1024 * 1024)

print(f"Original file size: {original_size_mb:.2f} MB ({format_size(original_size)})")
print()

# Test zstd compression directly
print("Testing zstd compression (level 9)...")
result = subprocess.run(
    ["zstd", "-c", "-9", test_file],
    capture_output=True,
    check=True
)
compressed_size = len(result.stdout)
compressed_size_mb = compressed_size / (1024 * 1024)
compression_ratio = original_size / compressed_size

print(f"Zstd compressed size: {compressed_size_mb:.2f} MB ({format_size(compressed_size)})")
print(f"Compression ratio: {compression_ratio:.2f}x")
print()

# Estimate chunking overhead
# With 32KB average chunks
chunks = original_size // 32768
header_overhead = chunks * 44  # 44 bytes per chunk header
header_overhead_kb = header_overhead / 1024

print("Estimated chunking:")
print(f"  Chunks: ~{chunks} (32KB average)")
print(f"  Header overhead: {header_overhead_kb:.2f} KB")
print()

# Estimate total pack size
total_pack_size = compressed_size + header_overhead
total_pack_size_mb = total_pack_size / (1024 * 1024)

print("Estimated total pack size:")
print(f"  Compressed data: {compressed_size_mb:.2f} MB")
print(f"  + Headers: {header_overhead_kb:.2f} KB")
print(f"  = Total: {total_pack_size_mb:.2f} MB ({format_size(total_pack_size)})")
print()

# Estimate manifest size
# Each chunk hash = 32 bytes
manifest_hashes = chunks * 32
manifest_metadata = 2048  # ~2KB for metadata
manifest_total = manifest_hashes + manifest_metadata
manifest_kb = manifest_total / 1024

print("Estimated manifest size:")
print(f"  Chunk hashes: {manifest_hashes/1024:.2f} KB")
print(f"  Metadata: {manifest_metadata/1024:.2f} KB")
print(f"  Total: {manifest_kb:.2f} KB")
print()

print("Summary for 30MB folder:")
print(f"  Original: {original_size_mb:.2f} MB")
print(f"  Pack (.pkg): ~{total_pack_size_mb:.2f} MB (with deduplication, could be smaller)")
print(f"  Manifest: ~{manifest_kb:.2f} KB")
print()

print(f"💡 If this folder is already in packs (deduplicated), the manifest alone")
print(f"   is enough to reconstruct it - just {manifest_kb:.2f} KB instead of {original_size_mb:.2f} MB!")
print()

# Show space savings
space_saved = original_size - total_pack_size
space_saved_mb = space_saved / (1024 * 1024)
print(f"Space saved: {space_saved_mb:.2f} MB ({space_saved/original_size*100:.1f}% reduction)")

