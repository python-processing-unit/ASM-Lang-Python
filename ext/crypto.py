"""ASM-Lang extension: cryptography utilities (RSA, AES-GCM, AES-CBC).

Operators are namespaced under the "crypto." prefix when the module is loaded.
"""

from __future__ import annotations

import os
from typing import Any, List, Optional

import numpy as np

from extensions import ASMExtensionError, ExtensionAPI

ASM_LANG_EXTENSION_NAME = "crypto"
ASM_LANG_EXTENSION_API_VERSION = 1
ASM_LANG_EXTENSION_ASMODULE = True

try:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    from cryptography.hazmat.primitives import hashes, padding as sym_padding, serialization
    from cryptography.hazmat.primitives.asymmetric import padding as asym_padding, rsa
except Exception as exc:
    _CRYPTO_IMPORT_ERROR: Optional[Exception] = exc
else:
    _CRYPTO_IMPORT_ERROR = None


def _expect_int(v: Any, rule: str, location: Any) -> int:
    from interpreter import ASMRuntimeError, TYPE_INT

    if getattr(v, "type", None) != TYPE_INT:
        raise ASMRuntimeError(f"{rule} expects INT", location=location, rewrite_rule=rule)
    return int(v.value)


def _expect_str(v: Any, rule: str, location: Any) -> str:
    from interpreter import ASMRuntimeError, TYPE_STR

    if getattr(v, "type", None) != TYPE_STR:
        raise ASMRuntimeError(f"{rule} expects STR", location=location, rewrite_rule=rule)
    return str(v.value)


def _bytes_to_tns(data: bytes):
    from interpreter import TYPE_INT, TYPE_TNS, Tensor, Value

    if not data:
        arr = np.array([Value(TYPE_INT, 0)], dtype=object)
        return Value(TYPE_TNS, Tensor(shape=[1], data=arr))
    arr = np.array([Value(TYPE_INT, b) for b in data], dtype=object)
    return Value(TYPE_TNS, Tensor(shape=[len(data)], data=arr))


def _tns_to_bytes(v: Any, rule: str, location: Any) -> bytes:
    from interpreter import ASMRuntimeError, TYPE_INT, TYPE_TNS, Tensor

    if getattr(v, "type", None) != TYPE_TNS or not isinstance(v.value, Tensor):
        raise ASMRuntimeError(f"{rule} expects TNS byte array", location=location, rewrite_rule=rule)
    tensor = v.value
    if len(tensor.shape) != 1:
        raise ASMRuntimeError(f"{rule} expects a 1D tensor", location=location, rewrite_rule=rule)
    out = bytearray()
    for entry in tensor.data.flat:
        if getattr(entry, "type", None) != TYPE_INT:
            raise ASMRuntimeError(f"{rule} tensor entries must be INT", location=location, rewrite_rule=rule)
        b = int(entry.value)
        if b < 0 or b > 255:
            raise ASMRuntimeError(f"{rule} tensor entries must be in [0,255]", location=location, rewrite_rule=rule)
        out.append(b)
    return bytes(out)


def _make_str(s: str):
    from interpreter import TYPE_STR, Value

    return Value(TYPE_STR, str(s))


def _make_keypair_tns(public_pem: str, private_pem: str):
    from interpreter import TYPE_STR, TYPE_TNS, Tensor, Value

    arr = np.array([Value(TYPE_STR, public_pem), Value(TYPE_STR, private_pem)], dtype=object)
    return Value(TYPE_TNS, Tensor(shape=[2], data=arr))


def _require_key_length(key: bytes, rule: str, location: Any) -> None:
    from interpreter import ASMRuntimeError

    if len(key) not in (16, 24, 32):
        raise ASMRuntimeError(f"{rule} key must be 16, 24, or 32 bytes", location=location, rewrite_rule=rule)


def _crypto_random(_interpreter, args, _arg_nodes, _env, location):
    from interpreter import ASMRuntimeError

    n = _expect_int(args[0], "CRYPTO_RANDOM", location)
    if n <= 0:
        raise ASMRuntimeError("CRYPTO_RANDOM expects a positive length", location=location, rewrite_rule="CRYPTO_RANDOM")
    return _bytes_to_tns(os.urandom(n))


def _hex_to_bytes(_interpreter, args, _arg_nodes, _env, location):
    from interpreter import ASMRuntimeError

    text = _expect_str(args[0], "CRYPTO_HEX_TO_BYTES", location)
    cleaned = "".join(ch for ch in text if not ch.isspace())
    if cleaned.startswith(("0x", "0X")):
        cleaned = cleaned[2:]
    if len(cleaned) == 0:
        raise ASMRuntimeError("CRYPTO_HEX_TO_BYTES requires at least 1 byte", location=location, rewrite_rule="CRYPTO_HEX_TO_BYTES")
    if len(cleaned) % 2 != 0:
        raise ASMRuntimeError("CRYPTO_HEX_TO_BYTES expects even-length hex", location=location, rewrite_rule="CRYPTO_HEX_TO_BYTES")
    try:
        data = bytes.fromhex(cleaned)
    except ValueError:
        raise ASMRuntimeError("CRYPTO_HEX_TO_BYTES expects valid hex", location=location, rewrite_rule="CRYPTO_HEX_TO_BYTES")
    return _bytes_to_tns(data)


def _bytes_to_hex(_interpreter, args, _arg_nodes, _env, location):
    data = _tns_to_bytes(args[0], "CRYPTO_BYTES_TO_HEX", location)
    return _make_str(data.hex())


def _aes_gcm_encrypt(_interpreter, args, _arg_nodes, _env, location):
    from interpreter import ASMRuntimeError

    key = _tns_to_bytes(args[0], "AES_GCM_ENCRYPT", location)
    _require_key_length(key, "AES_GCM_ENCRYPT", location)
    iv = _tns_to_bytes(args[1], "AES_GCM_ENCRYPT", location)
    if not iv:
        raise ASMRuntimeError("AES_GCM_ENCRYPT expects non-empty IV", location=location, rewrite_rule="AES_GCM_ENCRYPT")
    plaintext = _tns_to_bytes(args[2], "AES_GCM_ENCRYPT", location)
    aad = _tns_to_bytes(args[3], "AES_GCM_ENCRYPT", location) if len(args) > 3 else b""
    try:
        out = AESGCM(key).encrypt(iv, plaintext, aad)
    except Exception as exc:
        raise ASMRuntimeError(f"AES_GCM_ENCRYPT failed: {exc}", location=location, rewrite_rule="AES_GCM_ENCRYPT")
    return _bytes_to_tns(out)


def _aes_gcm_decrypt(_interpreter, args, _arg_nodes, _env, location):
    from interpreter import ASMRuntimeError

    key = _tns_to_bytes(args[0], "AES_GCM_DECRYPT", location)
    _require_key_length(key, "AES_GCM_DECRYPT", location)
    iv = _tns_to_bytes(args[1], "AES_GCM_DECRYPT", location)
    if not iv:
        raise ASMRuntimeError("AES_GCM_DECRYPT expects non-empty IV", location=location, rewrite_rule="AES_GCM_DECRYPT")
    ciphertext = _tns_to_bytes(args[2], "AES_GCM_DECRYPT", location)
    aad = _tns_to_bytes(args[3], "AES_GCM_DECRYPT", location) if len(args) > 3 else b""
    try:
        out = AESGCM(key).decrypt(iv, ciphertext, aad)
    except Exception as exc:
        raise ASMRuntimeError(f"AES_GCM_DECRYPT failed: {exc}", location=location, rewrite_rule="AES_GCM_DECRYPT")
    return _bytes_to_tns(out)


def _aes_cbc_encrypt(_interpreter, args, _arg_nodes, _env, location):
    from interpreter import ASMRuntimeError

    key = _tns_to_bytes(args[0], "AES_CBC_ENCRYPT", location)
    _require_key_length(key, "AES_CBC_ENCRYPT", location)
    iv = _tns_to_bytes(args[1], "AES_CBC_ENCRYPT", location)
    if len(iv) != 16:
        raise ASMRuntimeError("AES_CBC_ENCRYPT expects 16-byte IV", location=location, rewrite_rule="AES_CBC_ENCRYPT")
    plaintext = _tns_to_bytes(args[2], "AES_CBC_ENCRYPT", location)
    pad_flag = _expect_int(args[3], "AES_CBC_ENCRYPT", location) if len(args) > 3 else 1
    if pad_flag not in (0, 1):
        raise ASMRuntimeError("AES_CBC_ENCRYPT padding flag must be 0 or 1", location=location, rewrite_rule="AES_CBC_ENCRYPT")
    data = plaintext
    if pad_flag:
        padder = sym_padding.PKCS7(128).padder()
        data = padder.update(data) + padder.finalize()
    elif len(data) % 16 != 0:
        raise ASMRuntimeError("AES_CBC_ENCRYPT requires block-sized input when padding is disabled", location=location, rewrite_rule="AES_CBC_ENCRYPT")
    try:
        encryptor = Cipher(algorithms.AES(key), modes.CBC(iv)).encryptor()
        out = encryptor.update(data) + encryptor.finalize()
    except Exception as exc:
        raise ASMRuntimeError(f"AES_CBC_ENCRYPT failed: {exc}", location=location, rewrite_rule="AES_CBC_ENCRYPT")
    return _bytes_to_tns(out)


def _aes_cbc_decrypt(_interpreter, args, _arg_nodes, _env, location):
    from interpreter import ASMRuntimeError

    key = _tns_to_bytes(args[0], "AES_CBC_DECRYPT", location)
    _require_key_length(key, "AES_CBC_DECRYPT", location)
    iv = _tns_to_bytes(args[1], "AES_CBC_DECRYPT", location)
    if len(iv) != 16:
        raise ASMRuntimeError("AES_CBC_DECRYPT expects 16-byte IV", location=location, rewrite_rule="AES_CBC_DECRYPT")
    ciphertext = _tns_to_bytes(args[2], "AES_CBC_DECRYPT", location)
    if len(ciphertext) % 16 != 0:
        raise ASMRuntimeError("AES_CBC_DECRYPT expects ciphertext in 16-byte blocks", location=location, rewrite_rule="AES_CBC_DECRYPT")
    pad_flag = _expect_int(args[3], "AES_CBC_DECRYPT", location) if len(args) > 3 else 1
    if pad_flag not in (0, 1):
        raise ASMRuntimeError("AES_CBC_DECRYPT padding flag must be 0 or 1", location=location, rewrite_rule="AES_CBC_DECRYPT")
    try:
        decryptor = Cipher(algorithms.AES(key), modes.CBC(iv)).decryptor()
        data = decryptor.update(ciphertext) + decryptor.finalize()
        if pad_flag:
            unpadder = sym_padding.PKCS7(128).unpadder()
            data = unpadder.update(data) + unpadder.finalize()
    except Exception as exc:
        raise ASMRuntimeError(f"AES_CBC_DECRYPT failed: {exc}", location=location, rewrite_rule="AES_CBC_DECRYPT")
    return _bytes_to_tns(data)


def _rsa_keygen(_interpreter, args, _arg_nodes, _env, location):
    from interpreter import ASMRuntimeError

    bits = _expect_int(args[0], "RSA_KEYGEN", location)
    exponent = _expect_int(args[1], "RSA_KEYGEN", location) if len(args) > 1 else 65537
    if bits < 1024:
        raise ASMRuntimeError("RSA_KEYGEN requires at least 1024 bits", location=location, rewrite_rule="RSA_KEYGEN")
    try:
        key = rsa.generate_private_key(public_exponent=exponent, key_size=bits)
        priv_bytes = key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
        pub_bytes = key.public_key().public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    except Exception as exc:
        raise ASMRuntimeError(f"RSA_KEYGEN failed: {exc}", location=location, rewrite_rule="RSA_KEYGEN")
    return _make_keypair_tns(pub_bytes.decode("utf-8"), priv_bytes.decode("utf-8"))


def _rsa_public_from_private(_interpreter, args, _arg_nodes, _env, location):
    from interpreter import ASMRuntimeError

    priv_pem = _expect_str(args[0], "RSA_PUBLIC_FROM_PRIVATE", location)
    try:
        key = serialization.load_pem_private_key(priv_pem.encode("utf-8"), password=None)
        pub_bytes = key.public_key().public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    except Exception as exc:
        raise ASMRuntimeError(f"RSA_PUBLIC_FROM_PRIVATE failed: {exc}", location=location, rewrite_rule="RSA_PUBLIC_FROM_PRIVATE")
    return _make_str(pub_bytes.decode("utf-8"))


def _rsa_encrypt(_interpreter, args, _arg_nodes, _env, location):
    from interpreter import ASMRuntimeError

    pub_pem = _expect_str(args[0], "RSA_ENCRYPT", location)
    plaintext = _tns_to_bytes(args[1], "RSA_ENCRYPT", location)
    try:
        pub = serialization.load_pem_public_key(pub_pem.encode("utf-8"))
        out = pub.encrypt(
            plaintext,
            asym_padding.OAEP(
                mgf=asym_padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None,
            ),
        )
    except Exception as exc:
        raise ASMRuntimeError(f"RSA_ENCRYPT failed: {exc}", location=location, rewrite_rule="RSA_ENCRYPT")
    return _bytes_to_tns(out)


def _rsa_decrypt(_interpreter, args, _arg_nodes, _env, location):
    from interpreter import ASMRuntimeError

    priv_pem = _expect_str(args[0], "RSA_DECRYPT", location)
    ciphertext = _tns_to_bytes(args[1], "RSA_DECRYPT", location)
    try:
        priv = serialization.load_pem_private_key(priv_pem.encode("utf-8"), password=None)
        out = priv.decrypt(
            ciphertext,
            asym_padding.OAEP(
                mgf=asym_padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None,
            ),
        )
    except Exception as exc:
        raise ASMRuntimeError(f"RSA_DECRYPT failed: {exc}", location=location, rewrite_rule="RSA_DECRYPT")
    return _bytes_to_tns(out)


def asm_lang_register(ext: ExtensionAPI) -> None:
    if _CRYPTO_IMPORT_ERROR is not None:
        raise ASMExtensionError(
            f"crypto extension requires the 'cryptography' package ({_CRYPTO_IMPORT_ERROR})"
        )

    ext.metadata(name="crypto", version="0.1.0")
    ext.register_operator("CRYPTO_RANDOM", 1, 1, _crypto_random, doc="CRYPTO_RANDOM(n) -> TNS bytes")
    ext.register_operator("CRYPTO_HEX_TO_BYTES", 1, 1, _hex_to_bytes, doc="CRYPTO_HEX_TO_BYTES(hex) -> TNS bytes")
    ext.register_operator("CRYPTO_BYTES_TO_HEX", 1, 1, _bytes_to_hex, doc="CRYPTO_BYTES_TO_HEX(bytes) -> STR hex")
    ext.register_operator("AES_GCM_ENCRYPT", 3, 4, _aes_gcm_encrypt, doc="AES_GCM_ENCRYPT(key, iv, pt[, aad]) -> TNS (ct||tag)")
    ext.register_operator("AES_GCM_DECRYPT", 3, 4, _aes_gcm_decrypt, doc="AES_GCM_DECRYPT(key, iv, ct||tag[, aad]) -> TNS")
    ext.register_operator("AES_CBC_ENCRYPT", 3, 4, _aes_cbc_encrypt, doc="AES_CBC_ENCRYPT(key, iv, pt[, pad]) -> TNS")
    ext.register_operator("AES_CBC_DECRYPT", 3, 4, _aes_cbc_decrypt, doc="AES_CBC_DECRYPT(key, iv, ct[, pad]) -> TNS")
    ext.register_operator("RSA_KEYGEN", 1, 2, _rsa_keygen, doc="RSA_KEYGEN(bits[, exponent]) -> TNS [public_pem, private_pem]")
    ext.register_operator("RSA_PUBLIC_FROM_PRIVATE", 1, 1, _rsa_public_from_private, doc="RSA_PUBLIC_FROM_PRIVATE(private_pem) -> STR")
    ext.register_operator("RSA_ENCRYPT", 2, 2, _rsa_encrypt, doc="RSA_ENCRYPT(public_pem, plaintext) -> TNS")
    ext.register_operator("RSA_DECRYPT", 2, 2, _rsa_decrypt, doc="RSA_DECRYPT(private_pem, ciphertext) -> TNS")
