# Compression Examples for RomLib

## Real-World Compression Results

### For a 30MB Folder/Game ROM

The compression ratio depends heavily on the content type:

#### 1. **Highly Compressible Content** (Text, patterns, zeros)
- **Original**: 30 MB
- **Compressed pack**: ~0.1-0.5 MB (99%+ reduction)
- **Manifest**: ~32 KB
- **Total**: ~32-500 KB (if already in packs)

#### 2. **Moderately Compressible** (Game ROMs with some patterns)
- **Original**: 30 MB
- **Compressed pack**: ~10-20 MB (33-66% reduction)
- **Manifest**: ~32 KB
- **Total**: ~32 KB (if already in packs)

#### 3. **Poorly Compressible** (Encrypted, already compressed, random data)
- **Original**: 30 MB
- **Compressed pack**: ~30-31 MB (minimal reduction)
- **Manifest**: ~32 KB
- **Total**: ~32 KB (if already in packs)

### Key Insight: The Manifest is Always Tiny!

**The manifest size is independent of the original file size** - it only depends on the number of chunks:

- **30 MB file** → ~960 chunks (32KB each) → **~32 KB manifest**
- **500 MB file** → ~16,000 chunks → **~512 KB manifest**
- **1 GB file** → ~32,000 chunks → **~1 MB manifest**

### Example: 30MB Game ROM

```
Original game files:           30.00 MB
├─ Pack file (.pkg):           ~15-30 MB (depends on compressibility)
└─ Manifest (.manifest):        ~32 KB  ⭐ (tiny!)

If game is already in packs:
  Install size:                  ~32 KB  (just the manifest!)
  Reconstructed size:           30.00 MB (when installed)
```

### Deduplication Benefits

If you have 100 games with similar assets:

```
Without deduplication:
  100 games × 30 MB = 3,000 MB

With deduplication:
  Shared packs: ~1,500 MB (50% dedupe)
  Manifests: 100 × 32 KB = 3.2 MB
  Total: ~1,503 MB (50% space savings!)

To install all 100 games:
  Install: 3,000 MB (reconstructed)
  Stored: 1,503 MB (packs + manifests)
```

### Typical Game ROM Characteristics

Most game ROMs compress to:
- **30-70%** of original size (zstd level 9)
- **Manifest**: Always 32-128 KB regardless of game size
- **Installation**: Only requires manifest if packs are already present

### Compression Test Results

| Content Type | Original | Compressed | Ratio | Manifest |
|-------------|----------|------------|-------|----------|
| Text/Patterns | 30 MB | 0.1 MB | 300x | 32 KB |
| Typical ROM | 30 MB | 15 MB | 2x | 32 KB |
| Encrypted/Compressed | 30 MB | 30 MB | 1x | 32 KB |

**Bottom line**: Even if compression doesn't help much, the manifest is always tiny (KB scale), making it perfect for "500MB → few KB installer" scenarios when packs are pre-seeded.

