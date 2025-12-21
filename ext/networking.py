"""ASM-Lang extension: networking utilities.

This extension provides a pragmatic, stdlib-first set of networking operators:

- TCP (optionally TLS)
- UDP
- HTTP/HTTPS (urllib)
- FTP/FTPS (ftplib)
- SMTP/SMTPS (smtplib)

Notes
-----
- The interpreter is synchronous; all operations are blocking.
- Handles are returned as INT ids. Use the matching *_CLOSE operator.
- Payloads are usually STR (encoded/decoded with a configurable coding).
  Where binary matters, operators also exist that return/accept TNS byte arrays
  (1D tensor of INT in [0,255]).
"""

from __future__ import annotations

import asyncio
import io
import os
import socket
import ssl
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from extensions import ExtensionAPI


ASM_LANG_EXTENSION_NAME = "networking"
ASM_LANG_EXTENSION_API_VERSION = 1
ASM_LANG_EXTENSION_ASMODULE = True


# ---- Value helpers (import lazily inside operators to avoid cycles) ----


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


def _make_int(n: int):
    from interpreter import TYPE_INT, Value

    return Value(TYPE_INT, int(n))


def _make_str(s: str):
    from interpreter import TYPE_STR, Value

    return Value(TYPE_STR, str(s))


def _bytes_to_tns(data: bytes):
    from interpreter import TYPE_INT, TYPE_TNS, Tensor, Value

    # Keep shape at least 1 to match existing BYTES behavior.
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


def _normalize_encoding(coding: str) -> str:
    c = (coding or "UTF-8").strip().lower().replace("_", "-")
    if c in ("utf8", "utf-8"):
        return "utf-8"
    if c in ("utf16", "utf-16"):
        return "utf-16"
    if c in ("ascii",):
        return "ascii"
    if c in ("latin1", "latin-1"):
        return "latin-1"
    if c in ("ansi",):
        # Match interpreter WRITEFILE behavior.
        return "cp1252" if os.name == "nt" else "latin-1"
    return coding


def _ms_to_seconds(ms: int) -> Optional[float]:
    if ms <= 0:
        return None
    return ms / 1000.0


@dataclass
class _NetState:
    next_id: int = 1
    tcp: Dict[int, socket.socket] = field(default_factory=dict)
    udp: Dict[int, socket.socket] = field(default_factory=dict)


def _state(interpreter: Any) -> _NetState:
    st = getattr(interpreter, "_net_ext_state", None)
    if st is None:
        st = _NetState()
        setattr(interpreter, "_net_ext_state", st)
    return st


def _alloc_handle(st: _NetState) -> int:
    hid = st.next_id
    st.next_id += 1
    return hid


def _get_handle(map_: Dict[int, socket.socket], hid: int, rule: str, location: Any) -> socket.socket:
    from interpreter import ASMRuntimeError

    sock = map_.get(hid)
    if sock is None:
        raise ASMRuntimeError(f"{rule}: invalid handle", location=location, rewrite_rule=rule)
    return sock


def _ssl_context(*, verify: bool) -> ssl.SSLContext:
    if verify:
        return ssl.create_default_context()
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


# ---- TCP ----


def _tcp_connect(interpreter, args, _arg_nodes, _env, location):
    from interpreter import ASMRuntimeError

    host = _expect_str(args[0], "TCP_CONNECT", location)
    port = _expect_int(args[1], "TCP_CONNECT", location)

    timeout_ms = _expect_int(args[2], "TCP_CONNECT", location) if len(args) >= 3 else 5000
    tls = _expect_int(args[3], "TCP_CONNECT", location) if len(args) >= 4 else 0
    verify = _expect_int(args[4], "TCP_CONNECT", location) if len(args) >= 5 else 1
    server_hostname = _expect_str(args[5], "TCP_CONNECT", location) if len(args) >= 6 else host

    if port < 0 or port > 65535:
        raise ASMRuntimeError("TCP_CONNECT: port out of range", location=location, rewrite_rule="TCP_CONNECT")

    timeout_s = _ms_to_seconds(timeout_ms)

    try:
        raw = socket.create_connection((host, port), timeout=timeout_s)
        raw.settimeout(timeout_s)
        sock: socket.socket
        if tls != 0:
            ctx = _ssl_context(verify=verify != 0)
            sock = ctx.wrap_socket(raw, server_hostname=server_hostname or None)
        else:
            sock = raw
    except Exception as exc:
        raise ASMRuntimeError(f"TCP_CONNECT failed: {exc}", location=location, rewrite_rule="TCP_CONNECT")

    st = _state(interpreter)
    hid = _alloc_handle(st)
    st.tcp[hid] = sock
    return _make_int(hid)


def _tcp_send(interpreter, args, _arg_nodes, _env, location):
    from interpreter import ASMRuntimeError

    hid = _expect_int(args[0], "TCP_SEND", location)
    payload = _expect_str(args[1], "TCP_SEND", location)
    coding = _expect_str(args[2], "TCP_SEND", location) if len(args) >= 3 else "UTF-8"

    sock = _get_handle(_state(interpreter).tcp, hid, "TCP_SEND", location)
    enc = _normalize_encoding(coding)
    data = payload.encode(enc, errors="strict")

    try:
        sent = sock.send(data)
        return _make_int(sent)
    except Exception as exc:
        raise ASMRuntimeError(f"TCP_SEND failed: {exc}", location=location, rewrite_rule="TCP_SEND")


def _tcp_recv_text(interpreter, args, _arg_nodes, _env, location):
    from interpreter import ASMRuntimeError

    hid = _expect_int(args[0], "TCP_RECV_TEXT", location)
    max_bytes = _expect_int(args[1], "TCP_RECV_TEXT", location)
    coding = _expect_str(args[2], "TCP_RECV_TEXT", location) if len(args) >= 3 else "UTF-8"

    if max_bytes <= 0:
        raise ASMRuntimeError("TCP_RECV_TEXT: max_bytes must be > 0", location=location, rewrite_rule="TCP_RECV_TEXT")

    sock = _get_handle(_state(interpreter).tcp, hid, "TCP_RECV_TEXT", location)
    try:
        data = sock.recv(max_bytes)
    except Exception as exc:
        raise ASMRuntimeError(f"TCP_RECV_TEXT failed: {exc}", location=location, rewrite_rule="TCP_RECV_TEXT")

    enc = _normalize_encoding(coding)
    return _make_str(data.decode(enc, errors="replace"))


def _tcp_recv_bytes(interpreter, args, _arg_nodes, _env, location):
    from interpreter import ASMRuntimeError

    hid = _expect_int(args[0], "TCP_RECV_BYTES", location)
    max_bytes = _expect_int(args[1], "TCP_RECV_BYTES", location)

    if max_bytes <= 0:
        raise ASMRuntimeError("TCP_RECV_BYTES: max_bytes must be > 0", location=location, rewrite_rule="TCP_RECV_BYTES")

    sock = _get_handle(_state(interpreter).tcp, hid, "TCP_RECV_BYTES", location)
    try:
        data = sock.recv(max_bytes)
    except Exception as exc:
        raise ASMRuntimeError(f"TCP_RECV_BYTES failed: {exc}", location=location, rewrite_rule="TCP_RECV_BYTES")

    return _bytes_to_tns(data)


def _tcp_close(interpreter, args, _arg_nodes, _env, location):
    from interpreter import ASMRuntimeError

    hid = _expect_int(args[0], "TCP_CLOSE", location)
    st = _state(interpreter)
    sock = st.tcp.pop(hid, None)
    if sock is None:
        raise ASMRuntimeError("TCP_CLOSE: invalid handle", location=location, rewrite_rule="TCP_CLOSE")
    try:
        sock.close()
    except Exception:
        pass
    return _make_int(0)


# ---- UDP ----


def _udp_bind(interpreter, args, _arg_nodes, _env, location):
    from interpreter import ASMRuntimeError

    host = _expect_str(args[0], "UDP_BIND", location)
    port = _expect_int(args[1], "UDP_BIND", location)
    timeout_ms = _expect_int(args[2], "UDP_BIND", location) if len(args) >= 3 else 0

    if port < 0 or port > 65535:
        raise ASMRuntimeError("UDP_BIND: port out of range", location=location, rewrite_rule="UDP_BIND")

    timeout_s = _ms_to_seconds(timeout_ms)

    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        if timeout_s is not None:
            sock.settimeout(timeout_s)
        sock.bind((host, port))
    except Exception as exc:
        raise ASMRuntimeError(f"UDP_BIND failed: {exc}", location=location, rewrite_rule="UDP_BIND")

    st = _state(interpreter)
    hid = _alloc_handle(st)
    st.udp[hid] = sock
    return _make_int(hid)


def _udp_send(interpreter, args, _arg_nodes, _env, location):
    from interpreter import ASMRuntimeError

    hid = _expect_int(args[0], "UDP_SEND", location)
    host = _expect_str(args[1], "UDP_SEND", location)
    port = _expect_int(args[2], "UDP_SEND", location)
    payload = _expect_str(args[3], "UDP_SEND", location)
    coding = _expect_str(args[4], "UDP_SEND", location) if len(args) >= 5 else "UTF-8"

    if port < 0 or port > 65535:
        raise ASMRuntimeError("UDP_SEND: port out of range", location=location, rewrite_rule="UDP_SEND")

    sock = _get_handle(_state(interpreter).udp, hid, "UDP_SEND", location)
    enc = _normalize_encoding(coding)
    data = payload.encode(enc, errors="strict")
    try:
        sent = sock.sendto(data, (host, port))
        return _make_int(sent)
    except Exception as exc:
        raise ASMRuntimeError(f"UDP_SEND failed: {exc}", location=location, rewrite_rule="UDP_SEND")


def _udp_recv_text(interpreter, args, _arg_nodes, _env, location):
    from interpreter import ASMRuntimeError

    hid = _expect_int(args[0], "UDP_RECV_TEXT", location)
    max_bytes = _expect_int(args[1], "UDP_RECV_TEXT", location)
    timeout_ms = _expect_int(args[2], "UDP_RECV_TEXT", location) if len(args) >= 3 else 0
    coding = _expect_str(args[3], "UDP_RECV_TEXT", location) if len(args) >= 4 else "UTF-8"

    if max_bytes <= 0:
        raise ASMRuntimeError("UDP_RECV_TEXT: max_bytes must be > 0", location=location, rewrite_rule="UDP_RECV_TEXT")

    sock = _get_handle(_state(interpreter).udp, hid, "UDP_RECV_TEXT", location)
    timeout_s = _ms_to_seconds(timeout_ms)
    if timeout_s is not None:
        sock.settimeout(timeout_s)

    try:
        data, _addr = sock.recvfrom(max_bytes)
    except Exception as exc:
        raise ASMRuntimeError(f"UDP_RECV_TEXT failed: {exc}", location=location, rewrite_rule="UDP_RECV_TEXT")

    enc = _normalize_encoding(coding)
    return _make_str(data.decode(enc, errors="replace"))


def _udp_recv_bytes(interpreter, args, _arg_nodes, _env, location):
    from interpreter import ASMRuntimeError

    hid = _expect_int(args[0], "UDP_RECV_BYTES", location)
    max_bytes = _expect_int(args[1], "UDP_RECV_BYTES", location)
    timeout_ms = _expect_int(args[2], "UDP_RECV_BYTES", location) if len(args) >= 3 else 0

    if max_bytes <= 0:
        raise ASMRuntimeError("UDP_RECV_BYTES: max_bytes must be > 0", location=location, rewrite_rule="UDP_RECV_BYTES")

    sock = _get_handle(_state(interpreter).udp, hid, "UDP_RECV_BYTES", location)
    timeout_s = _ms_to_seconds(timeout_ms)
    if timeout_s is not None:
        sock.settimeout(timeout_s)

    try:
        data, _addr = sock.recvfrom(max_bytes)
    except Exception as exc:
        raise ASMRuntimeError(f"UDP_RECV_BYTES failed: {exc}", location=location, rewrite_rule="UDP_RECV_BYTES")

    return _bytes_to_tns(data)


def _udp_close(interpreter, args, _arg_nodes, _env, location):
    from interpreter import ASMRuntimeError

    hid = _expect_int(args[0], "UDP_CLOSE", location)
    st = _state(interpreter)
    sock = st.udp.pop(hid, None)
    if sock is None:
        raise ASMRuntimeError("UDP_CLOSE: invalid handle", location=location, rewrite_rule="UDP_CLOSE")
    try:
        sock.close()
    except Exception:
        pass
    return _make_int(0)


# ---- HTTP/HTTPS ----


def _http_request_bytes(method: str, url: str, *, body: Optional[bytes], content_type: Optional[str], timeout_ms: int, verify: bool) -> Tuple[int, bytes]:
    req = urllib.request.Request(url=url, data=body, method=method.upper())
    if content_type is not None:
        req.add_header("Content-Type", content_type)

    timeout_s = _ms_to_seconds(timeout_ms)
    ctx = _ssl_context(verify=verify)

    with urllib.request.urlopen(req, timeout=timeout_s, context=ctx) as resp:
        status = getattr(resp, "status", 200)
        data = resp.read()
        return int(status), data


def _http_get_text(interpreter, args, _arg_nodes, _env, location):
    from interpreter import ASMRuntimeError

    url = _expect_str(args[0], "HTTP_GET_TEXT", location)
    timeout_ms = _expect_int(args[1], "HTTP_GET_TEXT", location) if len(args) >= 2 else 5000
    verify = _expect_int(args[2], "HTTP_GET_TEXT", location) if len(args) >= 3 else 1

    try:
        _status, data = _http_request_bytes("GET", url, body=None, content_type=None, timeout_ms=timeout_ms, verify=verify != 0)
    except Exception as exc:
        raise ASMRuntimeError(f"HTTP_GET_TEXT failed: {exc}", location=location, rewrite_rule="HTTP_GET_TEXT")

    return _make_str(data.decode("utf-8", errors="replace"))


def _http_get_bytes(interpreter, args, _arg_nodes, _env, location):
    from interpreter import ASMRuntimeError

    url = _expect_str(args[0], "HTTP_GET_BYTES", location)
    timeout_ms = _expect_int(args[1], "HTTP_GET_BYTES", location) if len(args) >= 2 else 5000
    verify = _expect_int(args[2], "HTTP_GET_BYTES", location) if len(args) >= 3 else 1

    try:
        _status, data = _http_request_bytes("GET", url, body=None, content_type=None, timeout_ms=timeout_ms, verify=verify != 0)
    except Exception as exc:
        raise ASMRuntimeError(f"HTTP_GET_BYTES failed: {exc}", location=location, rewrite_rule="HTTP_GET_BYTES")

    return _bytes_to_tns(data)


def _http_get_status(interpreter, args, _arg_nodes, _env, location):
    from interpreter import ASMRuntimeError

    url = _expect_str(args[0], "HTTP_GET_STATUS", location)
    timeout_ms = _expect_int(args[1], "HTTP_GET_STATUS", location) if len(args) >= 2 else 5000
    verify = _expect_int(args[2], "HTTP_GET_STATUS", location) if len(args) >= 3 else 1

    try:
        status, _data = _http_request_bytes("GET", url, body=None, content_type=None, timeout_ms=timeout_ms, verify=verify != 0)
    except Exception as exc:
        raise ASMRuntimeError(f"HTTP_GET_STATUS failed: {exc}", location=location, rewrite_rule="HTTP_GET_STATUS")

    return _make_int(status)


def _http_post_text(interpreter, args, _arg_nodes, _env, location):
    from interpreter import ASMRuntimeError

    url = _expect_str(args[0], "HTTP_POST_TEXT", location)
    body_str = _expect_str(args[1], "HTTP_POST_TEXT", location)
    content_type = _expect_str(args[2], "HTTP_POST_TEXT", location) if len(args) >= 3 else "text/plain; charset=utf-8"
    timeout_ms = _expect_int(args[3], "HTTP_POST_TEXT", location) if len(args) >= 4 else 5000
    verify = _expect_int(args[4], "HTTP_POST_TEXT", location) if len(args) >= 5 else 1

    try:
        _status, data = _http_request_bytes(
            "POST",
            url,
            body=body_str.encode("utf-8"),
            content_type=content_type,
            timeout_ms=timeout_ms,
            verify=verify != 0,
        )
    except Exception as exc:
        raise ASMRuntimeError(f"HTTP_POST_TEXT failed: {exc}", location=location, rewrite_rule="HTTP_POST_TEXT")

    return _make_str(data.decode("utf-8", errors="replace"))


# ---- FTP / FTPS ----


def _ftp_login(host: str, port: int, user: str, password: str, *, tls: bool, timeout_s: Optional[float], verify: bool):
    import ftplib

    if tls:
        ftps = ftplib.FTP_TLS()
        ftps.context = ssl.create_default_context() if verify else _ssl_context(verify=False)
        ftps.connect(host=host, port=port, timeout=timeout_s)
        if user or password:
            ftps.login(user=user, passwd=password)
        else:
            ftps.login()
        ftps.prot_p()
        return ftps

    ftp = ftplib.FTP()
    ftp.connect(host=host, port=port, timeout=timeout_s)
    if user or password:
        ftp.login(user=user, passwd=password)
    else:
        ftp.login()
    return ftp


def _ftp_list(interpreter, args, _arg_nodes, _env, location):
    from interpreter import ASMRuntimeError

    host = _expect_str(args[0], "FTP_LIST", location)
    port = _expect_int(args[1], "FTP_LIST", location)
    user = _expect_str(args[2], "FTP_LIST", location)
    password = _expect_str(args[3], "FTP_LIST", location)
    directory = _expect_str(args[4], "FTP_LIST", location)
    tls = _expect_int(args[5], "FTP_LIST", location) if len(args) >= 6 else 0
    timeout_ms = _expect_int(args[6], "FTP_LIST", location) if len(args) >= 7 else 10000
    verify = _expect_int(args[7], "FTP_LIST", location) if len(args) >= 8 else 1

    if port < 0 or port > 65535:
        raise ASMRuntimeError("FTP_LIST: port out of range", location=location, rewrite_rule="FTP_LIST")

    timeout_s = _ms_to_seconds(timeout_ms)

    try:
        ftp = _ftp_login(host, port, user, password, tls=tls != 0, timeout_s=timeout_s, verify=verify != 0)
        lines: List[str] = []
        try:
            ftp.retrlines(f"LIST {directory}", callback=lines.append)
        finally:
            try:
                ftp.quit()
            except Exception:
                ftp.close()
        return _make_str("\n".join(lines))
    except Exception as exc:
        raise ASMRuntimeError(f"FTP_LIST failed: {exc}", location=location, rewrite_rule="FTP_LIST")


def _ftp_get_bytes(interpreter, args, _arg_nodes, _env, location):
    from interpreter import ASMRuntimeError

    host = _expect_str(args[0], "FTP_GET_BYTES", location)
    port = _expect_int(args[1], "FTP_GET_BYTES", location)
    user = _expect_str(args[2], "FTP_GET_BYTES", location)
    password = _expect_str(args[3], "FTP_GET_BYTES", location)
    path = _expect_str(args[4], "FTP_GET_BYTES", location)
    tls = _expect_int(args[5], "FTP_GET_BYTES", location) if len(args) >= 6 else 0
    timeout_ms = _expect_int(args[6], "FTP_GET_BYTES", location) if len(args) >= 7 else 10000
    verify = _expect_int(args[7], "FTP_GET_BYTES", location) if len(args) >= 8 else 1

    if port < 0 or port > 65535:
        raise ASMRuntimeError("FTP_GET_BYTES: port out of range", location=location, rewrite_rule="FTP_GET_BYTES")

    timeout_s = _ms_to_seconds(timeout_ms)

    try:
        ftp = _ftp_login(host, port, user, password, tls=tls != 0, timeout_s=timeout_s, verify=verify != 0)
        buf = io.BytesIO()
        try:
            ftp.retrbinary(f"RETR {path}", callback=buf.write)
        finally:
            try:
                ftp.quit()
            except Exception:
                ftp.close()
        return _bytes_to_tns(buf.getvalue())
    except Exception as exc:
        raise ASMRuntimeError(f"FTP_GET_BYTES failed: {exc}", location=location, rewrite_rule="FTP_GET_BYTES")


def _ftp_put_bytes(interpreter, args, _arg_nodes, _env, location):
    from interpreter import ASMRuntimeError

    host = _expect_str(args[0], "FTP_PUT_BYTES", location)
    port = _expect_int(args[1], "FTP_PUT_BYTES", location)
    user = _expect_str(args[2], "FTP_PUT_BYTES", location)
    password = _expect_str(args[3], "FTP_PUT_BYTES", location)
    path = _expect_str(args[4], "FTP_PUT_BYTES", location)
    payload = args[5]
    tls = _expect_int(args[6], "FTP_PUT_BYTES", location) if len(args) >= 7 else 0
    timeout_ms = _expect_int(args[7], "FTP_PUT_BYTES", location) if len(args) >= 8 else 10000
    verify = _expect_int(args[8], "FTP_PUT_BYTES", location) if len(args) >= 9 else 1

    if port < 0 or port > 65535:
        raise ASMRuntimeError("FTP_PUT_BYTES: port out of range", location=location, rewrite_rule="FTP_PUT_BYTES")

    data = _tns_to_bytes(payload, "FTP_PUT_BYTES", location)
    timeout_s = _ms_to_seconds(timeout_ms)

    try:
        ftp = _ftp_login(host, port, user, password, tls=tls != 0, timeout_s=timeout_s, verify=verify != 0)
        try:
            ftp.storbinary(f"STOR {path}", io.BytesIO(data))
        finally:
            try:
                ftp.quit()
            except Exception:
                ftp.close()
        return _make_int(1)
    except Exception as exc:
        raise ASMRuntimeError(f"FTP_PUT_BYTES failed: {exc}", location=location, rewrite_rule="FTP_PUT_BYTES")


# ---- SMTP / SMTPS ----


def _smtp_send(interpreter, args, _arg_nodes, _env, location):
    from interpreter import ASMRuntimeError

    host = _expect_str(args[0], "SMTP_SEND", location)
    port = _expect_int(args[1], "SMTP_SEND", location)
    user = _expect_str(args[2], "SMTP_SEND", location)
    password = _expect_str(args[3], "SMTP_SEND", location)
    mail_from = _expect_str(args[4], "SMTP_SEND", location)
    mail_to_raw = _expect_str(args[5], "SMTP_SEND", location)
    subject = _expect_str(args[6], "SMTP_SEND", location)
    body = _expect_str(args[7], "SMTP_SEND", location)
    tls = _expect_int(args[8], "SMTP_SEND", location) if len(args) >= 9 else 1
    timeout_ms = _expect_int(args[9], "SMTP_SEND", location) if len(args) >= 10 else 10000
    verify = _expect_int(args[10], "SMTP_SEND", location) if len(args) >= 11 else 1

    if port < 0 or port > 65535:
        raise ASMRuntimeError("SMTP_SEND: port out of range", location=location, rewrite_rule="SMTP_SEND")

    recipients = [r.strip() for r in mail_to_raw.replace(";", ",").split(",") if r.strip()]
    if not recipients:
        raise ASMRuntimeError("SMTP_SEND: no recipients", location=location, rewrite_rule="SMTP_SEND")

    timeout_s = _ms_to_seconds(timeout_ms)

    msg_lines = [
        f"From: {mail_from}",
        f"To: {', '.join(recipients)}",
        f"Subject: {subject}",
        "MIME-Version: 1.0",
        "Content-Type: text/plain; charset=utf-8",
        "",
        body,
        "",
    ]
    msg = "\r\n".join(msg_lines)

    try:
        import smtplib

        if tls != 0:
            ctx = _ssl_context(verify=verify != 0)
            client = smtplib.SMTP_SSL(host=host, port=port, timeout=timeout_s, context=ctx)
        else:
            client = smtplib.SMTP(host=host, port=port, timeout=timeout_s)

        try:
            client.ehlo()
            if tls == 0:
                # Optional STARTTLS if caller used plain SMTP port and wants upgrade.
                # Not enabled automatically to avoid surprises.
                pass
            if user or password:
                client.login(user, password)
            client.sendmail(mail_from, recipients, msg)
        finally:
            try:
                client.quit()
            except Exception:
                try:
                    client.close()
                except Exception:
                    pass
        return _make_int(1)
    except Exception as exc:
        raise ASMRuntimeError(f"SMTP_SEND failed: {exc}", location=location, rewrite_rule="SMTP_SEND")


def asm_lang_register(ext: ExtensionAPI) -> None:
    ext.metadata(name="networking", version="0.1.0")

    # TCP
    ext.register_operator("TCP_CONNECT", 2, 6, _tcp_connect, doc="TCP_CONNECT(host, port[, timeout_ms[, tls[, verify[, server_hostname]]]]) -> handle")
    ext.register_operator("TCP_SEND", 2, 3, _tcp_send, doc="TCP_SEND(handle, text[, coding]) -> bytes_sent")
    ext.register_operator("TCP_RECV_TEXT", 2, 3, _tcp_recv_text, doc="TCP_RECV_TEXT(handle, max_bytes[, coding]) -> STR")
    ext.register_operator("TCP_RECV_BYTES", 2, 2, _tcp_recv_bytes, doc="TCP_RECV_BYTES(handle, max_bytes) -> TNS")
    ext.register_operator("TCP_CLOSE", 1, 1, _tcp_close, doc="TCP_CLOSE(handle) -> 0")

    # UDP
    ext.register_operator("UDP_BIND", 2, 3, _udp_bind, doc="UDP_BIND(host, port[, timeout_ms]) -> handle")
    ext.register_operator("UDP_SEND", 4, 5, _udp_send, doc="UDP_SEND(handle, host, port, text[, coding]) -> bytes_sent")
    ext.register_operator("UDP_RECV_TEXT", 2, 4, _udp_recv_text, doc="UDP_RECV_TEXT(handle, max_bytes[, timeout_ms[, coding]]) -> STR")
    ext.register_operator("UDP_RECV_BYTES", 2, 3, _udp_recv_bytes, doc="UDP_RECV_BYTES(handle, max_bytes[, timeout_ms]) -> TNS")
    ext.register_operator("UDP_CLOSE", 1, 1, _udp_close, doc="UDP_CLOSE(handle) -> 0")

    # HTTP/HTTPS
    ext.register_operator("HTTP_GET_TEXT", 1, 3, _http_get_text, doc="HTTP_GET_TEXT(url[, timeout_ms[, verify]]) -> STR")
    ext.register_operator("HTTP_GET_BYTES", 1, 3, _http_get_bytes, doc="HTTP_GET_BYTES(url[, timeout_ms[, verify]]) -> TNS")
    ext.register_operator("HTTP_GET_STATUS", 1, 3, _http_get_status, doc="HTTP_GET_STATUS(url[, timeout_ms[, verify]]) -> INT")
    ext.register_operator("HTTP_POST_TEXT", 2, 5, _http_post_text, doc="HTTP_POST_TEXT(url, body[, content_type[, timeout_ms[, verify]]]) -> STR")

    # FTP / FTPS
    ext.register_operator("FTP_LIST", 5, 8, _ftp_list, doc="FTP_LIST(host, port, user, pass, dir[, tls[, timeout_ms[, verify]]]) -> STR")
    ext.register_operator("FTP_GET_BYTES", 5, 8, _ftp_get_bytes, doc="FTP_GET_BYTES(host, port, user, pass, path[, tls[, timeout_ms[, verify]]]) -> TNS")
    ext.register_operator("FTP_PUT_BYTES", 6, 9, _ftp_put_bytes, doc="FTP_PUT_BYTES(host, port, user, pass, path, data_tns[, tls[, timeout_ms[, verify]]]) -> INT")

    # SMTP / SMTPS
    ext.register_operator("SMTP_SEND", 8, 11, _smtp_send, doc="SMTP_SEND(host, port, user, pass, from, to_csv, subject, body[, tls[, timeout_ms[, verify]]]) -> 1")
    # The request mentioned "MTP"; treat it as an alias for SMTP_SEND for now.
    ext.register_operator("MTP_SEND", 8, 11, _smtp_send, doc="Alias for SMTP_SEND")
