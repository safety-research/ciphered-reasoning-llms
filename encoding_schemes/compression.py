import gzip
import base64


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
