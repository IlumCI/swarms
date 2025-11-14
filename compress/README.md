# RomLib - Game Library Packstore System

A content-addressed packstore system for managing game ROM/ISO collections with deduplication and compact manifests. This system enables storing large game libraries efficiently by chunking, compressing, and deduplicating content across games, while each game installation is just a tiny manifest file.

## Features

- **Content-Defined Chunking (CDC)**: Uses Rabin fingerprinting for variable-sized chunks
- **Deduplication**: Identical chunks are stored once, referenced by BLAKE3 hash
- **Compression**: Per-chunk zstd compression with optional dictionary support
- **Compact Manifests**: Protobuf-based manifests that only reference chunk hashes
- **Local Cache**: LMDB-based index for fast hash-to-pack lookups
- **HTTP Server**: Local daemon for serving blobs via HTTP
- **Atomic Operations**: Safe file reconstruction with atomic moves
- **Cross-Platform**: Works on Linux, macOS, and Windows

## Requirements

- Rust 1.70+ (with Cargo)
- Protocol Buffers compiler (`protoc`)
  - Debian/Ubuntu: `apt-get install protobuf-compiler`
  - macOS: `brew install protobuf`
  - Or download from: https://github.com/protocolbuffers/protobuf/releases
- LMDB development libraries
  - Debian/Ubuntu: `apt-get install liblmdb-dev`
  - macOS: `brew install lmdb`

## Building

```bash
cargo build --release
```

## Directory Layout

On-device layout:

```
/romlib/              # read-only library
    packs/            # compressed shared content
        pack-0001.pkg
        pack-0002.pkg
        ...
    index.db          # key-value index (hash -> pack_id, offset, length)
    games/             # tiny per-game manifests ("tickets")
        game-123.manifest
        game-456.manifest

/install/             # install area (decompressed ROMs)
    title-id-1/
    title-id-2/
```

## Usage

### Build Packs from Raw Games

Process all games and create packs with deduplication:

```bash
romlib build-packs \
  --input raw_games/ \
  --output romlib/ \
  --compression-level 9
```

This creates:
- `romlib/packs/pack-0001.pkg` - Compressed chunks
- `romlib/index.db` - Hash-to-pack mapping

### Generate Game Manifests

Generate tiny manifest files for each game:

```bash
romlib generate-manifests \
  --input raw_games/ \
  --output romlib/
```

This creates `romlib/games/*.manifest` files (typically KB to hundreds of KB).

### List Available Games

List all games available from manifests:

```bash
romlib list-games --romlib romlib/
```

### Install a Game

Install a game from its manifest (reconstructs ROMs from packs):

```bash
romlib install \
  --game GAME001 \
  --romlib romlib/ \
  --install-dir /install
```

This reconstructs the game files to `/install/GAME001/` using chunks from packs.

### Uninstall a Game

Remove an installed game (packs remain untouched):

```bash
romlib uninstall --title-id GAME001 --install-dir /install
```

## Architecture

### Pack File Format (.pkg)

- **Header**: Magic number (`PKGSTR`), version, flags (32 bytes)
- **Chunks**: Sequence of compressed chunks with headers
- **Footer**: Checksum and optional signature

Each chunk contains:
- BLAKE3 hash (32 bytes)
- Uncompressed length (4 bytes)
- Compressed length (4 bytes)
- Compression method (1 byte)
- Flags (1 byte)
- Compressed payload

### Index Database

LMDB-based index mapping:
- Key: BLAKE3 hash (32 bytes)
- Value: Pack ID, offset, lengths, compression method, timestamp, refcount

### Manifest Format

Protobuf-based manifest containing:
- Package ID and version
- File entries with chunk hash sequences
- Required pack IDs
- Optional signature
- Metadata

## CLI Commands

### `build-packs`

Build packs from raw game ROMs/ISOs with deduplication.

Options:
- `--input, -i`: Input directory containing raw games
- `--output, -o`: Output romlib directory
- `--avg-chunk`: Average chunk size in bytes (default: 32768)
- `--compression-level, -c`: Compression level 1-22 (default: 9)
- `--dict, -d`: Optional dictionary file for compression

### `generate-manifests`

Generate manifests for all games (after building packs).

Options:
- `--input, -i`: Input directory (same as build-packs input)
- `--output, -o`: Output romlib directory (must contain packs/ and index.db)
- `--compression-level, -c`: Compression level (must match build-packs)

### `list-games`

List available games from manifests.

Options:
- `--romlib, -r`: Romlib directory containing games/ subdirectory

### `install`

Install a game from its manifest.

Options:
- `--game, -g`: Title ID or manifest path
- `--romlib, -r`: Romlib directory (contains packs/ and index.db)
- `--install-dir`: Install directory (default: `./install`)

### `uninstall`

Uninstall a game (removes install directory only).

Options:
- `--title-id, -t`: Title ID
- `--install-dir`: Install directory (default: `./install`)

## Performance Considerations

- **Chunk Size**: Default 8-64 KiB chunks balance deduplication effectiveness with overhead
- **Compression**: Level 3 provides good balance; higher levels for better compression
- **Index**: LMDB provides sub-millisecond lookups with memory-mapped I/O
- **Pack Files**: Use mmap for fast random access to chunks

## Workflow Example

1. **Prepare raw games**:
   ```
   raw_games/
     game-001/
       main.rom
     game-002/
       main.iso
   ```

2. **Build packs** (chunks and deduplicates all games):
   ```bash
   romlib build-packs --input raw_games/ --output romlib/
   ```

3. **Generate manifests** (creates tiny manifest per game):
   ```bash
   romlib generate-manifests --input raw_games/ --output romlib/
   ```

4. **Ship to device**: Just copy `romlib/` directory (packs + index.db + games/*.manifest)

5. **On device, install games**:
   ```bash
   romlib install --game GAME001 --romlib /romlib --install-dir /install
   ```

## Security

- Manifests can be signed with Ed25519 (signature verification not yet implemented)
- Pack files include checksums for integrity verification
- All operations are offline - no network required

## License

This project is provided as-is for demonstration purposes.

