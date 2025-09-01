import gzip
import base64
import io

def percent_gzip_bytestream(data: bytes) -> float:
    """
    Return the percentage (0..100) of bytes in `data` that form a valid gzip bytestream.

    - Accepts raw bytes instead of strings.
    - Scans for gzip members by detecting the magic header (0x1F, 0x8B).
    - Attempts to decompress from that position; if valid, marks that range as "covered".
    - Supports concatenated gzip members.
    - Ignores corrupted or partial members.
    """
    n_bytes = len(data)
    if n_bytes == 0:
        return 0.0

    covered = bytearray(n_bytes)  # 0/1 flags per byte position

    i = 0
    end = n_bytes
    while i < end - 1:
        # Check for gzip magic header
        if data[i] == 0x1F and data[i + 1] == 0x8B:
            bio = io.BytesIO(data[i:])
            try:
                with gzip.GzipFile(fileobj=bio) as gf:
                    # Drain the gzip member fully
                    while gf.read(1024 * 1024):
                        pass
                member_len = bio.tell()
                if member_len > 0:
                    # Mark covered bytes
                    end_idx = min(i + member_len, n_bytes)
                    covered[i:end_idx] = b"\x01" * (end_idx - i)
                    i += member_len
                    continue
            except (OSError, EOFError):
                # Invalid gzip starting at this position, move on
                pass
        i += 1

    covered_count = int(sum(covered))
    return covered_count / n_bytes



def bytes_to_unicode():
    # Printable bytes that we keep as-is:
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)  # map remaining bytes to private Unicode area
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


BYTE_TO_UNI = bytes_to_unicode()
UNI_TO_BYTE = {v: k for k, v in BYTE_TO_UNI.items()}


def bytes_to_shim_text(b: bytes) -> str:
    return "".join(BYTE_TO_UNI[x] for x in b)


def shim_text_to_bytes(s: str) -> bytes:
    return bytes(UNI_TO_BYTE[ch] for ch in s)


def gzip_to_bpe_encoded(s):
    gz_bytes = gzip.compress(s.encode("utf-8"))
    shim_text = bytes_to_shim_text(gz_bytes)

    return shim_text


def gzip_to_base64_encoded(s):
    gz_bytes = gzip.compress(s.encode("utf-8"))
    return base64.b64encode(gz_bytes).decode("ascii")


_BASE64_ALPHABET = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=")

def _b64_to_bytes_best_effort(s: str) -> bytes:
    """Salvage as many bytes as possible from a noisy Base64 string."""
    # Keep only legal Base64 symbols
    filtered = "".join(c for c in s if c in _BASE64_ALPHABET)

    # Fast path: try decoding the whole cleaned string (pad to /4)
    whole = _try_b64_decode(filtered)
    if whole is not None:
        return whole

    # Sliding salvage: decode valid quartets, skip past bad spots
    out = bytearray()
    i = 0
    n = len(filtered)
    while i < n:
        chunk = filtered[i:i+4]
        if not chunk:
            break
        padded = _pad4(chunk)
        try:
            out.extend(base64.b64decode(padded, validate=False))
            i += 4  # good quartet
        except Exception:
            i += 1  # skip one char and try again
    return bytes(out)

def _pad4(t: str) -> str:
    need = (-len(t)) % 4
    return t + ("=" * need if need else "")

def _try_b64_decode(t: str):
    try:
        return base64.b64decode(_pad4(t), validate=False)
    except Exception:
        return None


def _bpe_to_bytes_best_effort(s: str) -> bytes:
    """
    Convert 'shim text' back to bytes, skipping any characters that
    aren't present in the UNI_TO_BYTE mapping.
    """
    out = bytearray()
    for ch in s:
        b = UNI_TO_BYTE.get(ch)
        if b is not None:
            out.append(b)
        # else: drop invalid char
    return bytes(out)


def _gunzip_best_effort(data: bytes) -> bytes:
    """
    Best-effort gzip decompression:
    - Finds each gzip member (header 1F 8B 08) and tries to decompress from there.
    - On success, appends decompressed bytes and continues with the remainder.
    - On failure, advances one byte and tries again.
    Returns concatenated decompressed bytes (may be empty).
    """
    out = bytearray()
    buf = data
    while True:
        # Search for gzip header
        start = buf.find(b"\x1f\x8b\x08")
        if start == -1:
            break
        # Try to decompress from the header
        try:
            d = zlib.decompressobj(wbits=16 + zlib.MAX_WBITS)  # gzip wrapper
            chunk = d.decompress(buf[start:])
            out.extend(chunk)
            tail = d.unused_data
            if not tail:
                # No more data beyond this gzip member
                break
            # Continue scanning in the unused tail
            buf = tail
        except Exception:
            # Skip past this byte and keep looking
            buf = buf[start + 1:]
            if not buf:
                break
    return bytes(out)
# ===========================================================================


# ============================= public decoders ==============================
def inverse_gzip_to_bpe_encoded(s: str) -> str:
    """
    Inverse of gzip_to_bpe_encoded, best-effort.
    - Skips unmapped characters in the BPE shim text.
    - Decompresses any valid gzip members it can find.
    Returns UTF-8 text with replacement for undecodable bytes.
    """
    gz_bytes = _bpe_to_bytes_best_effort(s)
    plain_bytes = _gunzip_best_effort(gz_bytes)
    return plain_bytes.decode("utf-8", errors="replace")


def inverse_gzip_to_base64_encoded(s: str) -> str:
    """
    Inverse of gzip_to_base64_encoded, best-effort.
    - Discards invalid Base64 symbols and repairs padding.
    - Decodes salvageable quartets.
    - Decompresses any valid gzip members it can find.
    Returns UTF-8 text with replacement for undecodable bytes.
    """
    gz_bytes = _b64_to_bytes_best_effort(s)
    plain_bytes = _gunzip_best_effort(gz_bytes)
    return plain_bytes.decode("utf-8", errors="replace")


def calculate_gzip_base64_adherence(s):
    gz_bytes = _b64_to_bytes_best_effort(s)
    return percent_gzip_bytestream(gz_bytes) >= 0.5