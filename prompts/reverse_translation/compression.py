reverse_gzip_to_bpe_encoded_prompt = """
The input has been formatted such that the UTF-8 bytes of the input are gzipped at gzip level 9, and then converted to Unicode text
according to the bytes_to_shim_text function below.

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

Do not output anything except your translation.
"""

reverse_gzip_to_base64_encoded_prompt = """
The input has been formatted such that the UTF-8 bytes of the input are gzipped at gzip level 9, and then converted to base64 text.

Do not output anything except your translation.
"""