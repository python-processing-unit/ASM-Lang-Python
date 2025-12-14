from __future__ import annotations
import json
import subprocess
import math
import os
import sys
import platform
import tempfile
import codecs
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from numpy.typing import NDArray

from lexer import ASMError, ASMParseError, Lexer
from parser import (
    Assignment,
    Block,
    BreakStatement,
    CallArgument,
    CallExpression,
    Expression,
    ExpressionStatement,
    ForStatement,
    FuncDef,
    GotoStatement,
    GotopointStatement,
    Identifier,
    IndexExpression,
    IfBranch,
    IfStatement,
    Literal,
    Param,
    Parser,
    Program,
    ReturnStatement,
    PopStatement,
    SourceLocation,
    Statement,
    TensorLiteral,
    TensorSetStatement,
    WhileStatement,
    ContinueStatement,
)


TYPE_INT = "INT"
TYPE_STR = "STR"
TYPE_TNS = "TNS"

# On Windows, command lines over a certain length cause CreateProcess errors
WINDOWS_COMMAND_LENGTH_LIMIT = 8000

@dataclass(frozen=True)
class Tensor:
    shape: List[int]
    data: NDArray[Any]


@dataclass
class Value:
    type: str
    value: Union[int, str, Tensor]


class ASMRuntimeError(ASMError):
    """Raised for runtime faults."""

    def __init__(
        self,
        message: str,
        *,
        location: Optional[SourceLocation] = None,
        rewrite_rule: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.location = location
        self.rewrite_rule = rewrite_rule
        self.step_index: Optional[int] = None


class ReturnSignal(Exception):
    def __init__(self, value: Value) -> None:
        super().__init__(value)
        self.value = value


class ExitSignal(Exception):
    def __init__(self, code: int = 0) -> None:
        super().__init__(code)
        self.code = code


class BreakSignal(Exception):
    def __init__(self, count: int) -> None:
        super().__init__(count)
        self.count = count


class ContinueSignal(Exception):
    def __init__(self) -> None:
        super().__init__()


class JumpSignal(Exception):
    def __init__(self, target: Value) -> None:
        super().__init__(target)
        self.target = target


@dataclass
class Environment:
    parent: Optional["Environment"] = None
    values: Dict[str, Value] = field(default_factory=dict)
    frozen: set = field(default_factory=set)
    permafrozen: set = field(default_factory=set)

    def _find_env(self, name: str) -> Optional["Environment"]:
        env: Optional[Environment] = self
        while env is not None:
            if name in env.values:
                return env
            env = env.parent
        return None

    def set(self, name: str, value: Value, declared_type: Optional[str] = None) -> None:
        env = self._find_env(name)
        if env is not None:
            existing = env.values[name]
            if name in env.frozen or name in env.permafrozen:
                raise ASMRuntimeError(
                    f"Identifier '{name}' is frozen and cannot be reassigned",
                    rewrite_rule="ASSIGN",
                )
            if declared_type and existing.type != declared_type:
                raise ASMRuntimeError(
                    f"Type mismatch for '{name}': previously declared as {existing.type}",
                    rewrite_rule="ASSIGN",
                )
            if existing.type != value.type:
                raise ASMRuntimeError(
                    f"Type mismatch for '{name}': expected {existing.type} but got {value.type}",
                    rewrite_rule="ASSIGN",
                )
            env.values[name] = value
            return

        if declared_type is None:
            raise ASMRuntimeError(
                f"Identifier '{name}' must be declared with a type before assignment",
                rewrite_rule="ASSIGN",
            )
        if declared_type != value.type:
            raise ASMRuntimeError(
                f"Assigned value type {value.type} does not match declaration {declared_type}",
                rewrite_rule="ASSIGN",
            )
        self.values[name] = value

    def get(self, name: str) -> Value:
        env = self._find_env(name)
        if env is not None:
            return env.values[name]
        raise ASMRuntimeError(f"Undefined identifier '{name}'", rewrite_rule="IDENT")

    def delete(self, name: str) -> None:
        env = self._find_env(name)
        if env is not None:
            if name in env.frozen or name in env.permafrozen:
                raise ASMRuntimeError(f"Identifier '{name}' is frozen and cannot be deleted", rewrite_rule="DEL")
            del env.values[name]
            return
        raise ASMRuntimeError(f"Cannot delete undefined identifier '{name}'", rewrite_rule="DEL")

    def has(self, name: str) -> bool:
        return self._find_env(name) is not None

    def snapshot(self) -> Dict[str, str]:
        def _render(val: Value) -> str:
            if val.type == TYPE_TNS and isinstance(val.value, Tensor):
                dims = ",".join(str(d) for d in val.value.shape)
                return f"{val.type}:[{dims}]"
            return f"{val.type}:{val.value}"

        return {k: _render(v) for k, v in self.values.items()}

    def freeze(self, name: str) -> None:
        env = self._find_env(name)
        if env is None:
            raise ASMRuntimeError(f"Cannot freeze undefined identifier '{name}'", rewrite_rule="FREEZE")
        env.frozen.add(name)

    def thaw(self, name: str) -> None:
        env = self._find_env(name)
        if env is None:
            raise ASMRuntimeError(f"Cannot thaw undefined identifier '{name}'", rewrite_rule="THAW")
        if name in env.permafrozen:
            raise ASMRuntimeError(
                f"Identifier '{name}' is permanently frozen and cannot be thawed",
                rewrite_rule="THAW",
            )
        # silently succeed if not frozen
        env.frozen.discard(name)

    def permafreeze(self, name: str) -> None:
        env = self._find_env(name)
        if env is None:
            raise ASMRuntimeError(f"Cannot permafreeze undefined identifier '{name}'", rewrite_rule="PERMAFREEZE")
        env.frozen.add(name)
        env.permafrozen.add(name)


@dataclass
class Function:
    name: str
    params: List[Param]
    return_type: str
    body: Block
    closure: Environment


@dataclass
class Frame:
    name: str
    env: Environment
    frame_id: str
    call_location: Optional[SourceLocation]
    gotopoints: Dict[Any, int] = field(default_factory=dict)


@dataclass
class StateEntry:
    step_index: int
    state_id: str
    frame_id: Optional[str]
    source_location: Optional[SourceLocation]
    statement: Optional[str]
    env_snapshot: Optional[Dict[str, Any]]
    rewrite_record: Optional[Dict[str, Any]]


class StateLogger:
    def __init__(self, verbose: bool) -> None:
        self.verbose = verbose
        self.entries: List[StateEntry] = []
        self.next_state_index = 0
        self.last_state_id = "seed"
        self.frame_last_entry: Dict[str, StateEntry] = {}

    def record(
        self,
        *,
        frame: Optional[Frame],
        location: Optional[SourceLocation],
        statement: Optional[str],
        rewrite_record: Optional[Dict[str, Any]] = None,
        env_snapshot: Optional[Dict[str, int]] = None,
    ) -> StateEntry:
        rewrite = dict(rewrite_record or {})
        if "from_state_id" not in rewrite:
            rewrite["from_state_id"] = self.last_state_id
        step_index = self.next_state_index
        state_id = f"s_{step_index:06d}"
        rewrite["to_state_id"] = state_id
        entry = StateEntry(
            step_index=step_index,
            state_id=state_id,
            frame_id=frame.frame_id if frame else None,
            source_location=location,
            statement=statement,
            env_snapshot=env_snapshot,
            rewrite_record=rewrite,
        )
        self.entries.append(entry)
        if frame:
            self.frame_last_entry[frame.frame_id] = entry
        self.last_state_id = state_id
        self.next_state_index += 1
        return entry

    def last_entry_for_frame(self, frame_id: str) -> Optional[StateEntry]:
        return self.frame_last_entry.get(frame_id)


BuiltinImpl = Callable[["Interpreter", List[Value], List[Expression], Environment, SourceLocation], Value]


@dataclass
class BuiltinFunction:
    name: str
    min_args: int
    max_args: Optional[int]
    impl: BuiltinImpl

    def validate(self, supplied: int) -> None:
        if supplied < self.min_args:
            raise ASMRuntimeError(f"{self.name} expects at least {self.min_args} arguments", rewrite_rule=self.name)
        if self.max_args is not None and supplied > self.max_args:
            raise ASMRuntimeError(f"{self.name} expects at most {self.max_args} arguments", rewrite_rule=self.name)


def _as_bool(value: int) -> int:
    return 0 if value == 0 else 1


class Builtins:
    def __init__(self) -> None:
        self.table: Dict[str, BuiltinFunction] = {}
        self._register_int_only("ADD", 2, lambda a, b: a + b)
        self._register_int_only("SUB", 2, lambda a, b: a - b)
        self._register_int_only("MUL", 2, lambda a, b: a * b)
        self._register_int_only("DIV", 2, self._safe_div)
        self._register_int_only("CDIV", 2, self._safe_cdiv)
        self._register_int_only("MOD", 2, self._safe_mod)
        self._register_int_only("POW", 2, self._safe_pow)
        self._register_int_only("NEG", 1, lambda a: -a)
        self._register_int_only("ABS", 1, abs)
        self._register_int_only("GCD", 2, math.gcd)
        self._register_int_only("LCM", 2, self._lcm)
        self._register_int_only("BAND", 2, lambda a, b: a & b)
        self._register_int_only("BOR", 2, lambda a, b: a | b)
        self._register_int_only("BXOR", 2, lambda a, b: a ^ b)
        self._register_int_only("BNOT", 1, lambda a: ~a)
        self._register_int_only("SHL", 2, self._shift_left)
        self._register_int_only("SHR", 2, self._shift_right)
        self._register_custom("SLICE", 3, 3, self._slice)
        self._register_custom("AND", 2, 2, self._and)
        self._register_custom("OR", 2, 2, self._or)
        self._register_custom("XOR", 2, 2, self._xor)
        self._register_custom("NOT", 1, 1, self._not)
        self._register_custom("BOOL", 1, 1, self._bool)
        self._register_custom("ARGV", 0, 0, self._argv)
        self._register_custom("EQ", 2, 2, self._eq)
        self._register_custom("IN", 2, 2, self._in)
        self._register_int_only("GT", 2, lambda a, b: 1 if a > b else 0)
        self._register_int_only("LT", 2, lambda a, b: 1 if a < b else 0)
        self._register_int_only("GTE", 2, lambda a, b: 1 if a >= b else 0)
        self._register_int_only("LTE", 2, lambda a, b: 1 if a <= b else 0)
        self._register_variadic("SUM", 1, self._sum)
        self._register_variadic("PROD", 1, self._prod)
        self._register_variadic("MAX", 1, self._max)
        self._register_variadic("MIN", 1, self._min)
        self._register_variadic("ANY", 1, self._any)
        self._register_variadic("ALL", 1, self._all)
        self._register_variadic("LEN", 0, self._len)
        self._register_custom("SLEN", 1, 1, self._slen)
        self._register_custom("ILEN", 1, 1, self._ilen)
        self._register_variadic("JOIN", 1, self._join)
        self._register_custom("SPLIT", 1, 2, self._split)
        self._register_int_only("LOG", 1, self._safe_log)
        self._register_int_only("CLOG", 1, self._safe_clog)
        self._register_custom("INT", 1, 1, self._int_op)
        self._register_custom("STR", 1, 1, self._str_op)
        self._register_custom("UPPER", 1, 1, self._upper)
        self._register_custom("LOWER", 1, 1, self._lower)
        self._register_custom("STRIP", 2, 2, self._strip)
        self._register_custom("REPLACE", 3, 3, self._replace)
        self._register_custom("MAIN", 0, 0, self._main)
        self._register_custom("OS", 0, 0, self._os)
        self._register_custom("IMPORT", 1, 1, self._import)
        self._register_custom("RUN", 1, 1, self._run)
        self._register_custom("INPUT", 0, 1, self._input)
        self._register_custom("PRINT", 0, None, self._print)
        self._register_custom("ASSERT", 1, 1, self._assert)
        self._register_custom("DEL", 1, 1, self._delete)
        self._register_custom("FREEZE", 1, 1, self._freeze)
        self._register_custom("THAW", 1, 1, self._thaw)
        self._register_custom("PERMAFREEZE", 1, 1, self._permafreeze)
        self._register_custom("FROZEN", 1, 1, self._frozen)
        self._register_custom("PERMAFROZEN", 1, 1, self._permafrozen)
        self._register_custom("EXIST", 1, 1, self._exist)
        self._register_custom("EXPORT", 2, 2, self._export)
        self._register_custom("ISINT", 1, 1, self._isint)
        self._register_custom("ISSTR", 1, 1, self._isstr)
        self._register_custom("ISTNS", 1, 1, self._istns)
        self._register_custom("READFILE", 1, 2, self._readfile)
        self._register_custom("BYTES", 1, 1, self._bytes)
        self._register_custom("WRITEFILE", 2, 3, self._writefile)
        self._register_custom("EXISTFILE", 1, 1, self._existfile)
        self._register_custom("CL", 1, 1, self._cl)
        self._register_custom("EXIT", 0, 1, self._exit)
        self._register_custom("SHUSH", 0, 0, self._shush)
        self._register_custom("UNSHUSH", 0, 0, self._unshush)
        self._register_custom("SHAPE", 1, 1, self._shape)
        self._register_custom("TLEN", 2, 2, self._tlen)
        self._register_custom("FILL", 2, 2, self._fill)
        self._register_custom("TNS", 2, 2, self._tns)
        self._register_custom("MADD", 2, 2, self._madd)
        self._register_custom("MSUB", 2, 2, self._msub)
        self._register_custom("MMUL", 2, 2, self._mmul)
        self._register_custom("MDIV", 2, 2, self._mdiv)
        self._register_variadic("MSUM", 1, self._msum)
        self._register_variadic("MPROD", 1, self._mprod)
        self._register_custom("TADD", 2, 2, self._tadd)
        self._register_custom("TSUB", 2, 2, self._tsub)
        self._register_custom("TMUL", 2, 2, self._tmul)
        self._register_custom("TDIV", 2, 2, self._tdiv)
        self._register_custom("TPOW", 2, 2, self._tpow)

    def _register_int_only(self, name: str, arity: int, func: Callable[..., int]) -> None:
        def impl(_: "Interpreter", args: List[Value], __: List[Expression], ___: Environment, location: SourceLocation) -> Value:
            ints = [self._expect_int(arg, name, location) for arg in args]
            return Value(TYPE_INT, func(*ints))

        self.table[name] = BuiltinFunction(name=name, min_args=arity, max_args=arity, impl=impl)

    def _register_variadic(self, name: str, min_args: int, func: Callable[[List[Value], SourceLocation], Value]) -> None:
        def impl(_: "Interpreter", args: List[Value], __: List[Expression], ___: Environment, location: SourceLocation) -> Value:
            return func(args, location)

        self.table[name] = BuiltinFunction(name=name, min_args=min_args, max_args=None, impl=impl)

    def _register_custom(
        self,
        name: str,
        min_args: int,
        max_args: Optional[int],
        impl: BuiltinImpl,
    ) -> None:
        self.table[name] = BuiltinFunction(name=name, min_args=min_args, max_args=max_args, impl=impl)

    def invoke(
        self,
        interpreter: "Interpreter",
        name: str,
        args: List[Value],
        arg_nodes: List[Expression],
        env: Environment,
        location: SourceLocation,
    ) -> Value:
        builtin = self.table.get(name)
        if builtin is None:
            raise ASMRuntimeError(f"Unknown function '{name}'", location=location)
        supplied = len(args)
        if supplied < builtin.min_args:
            raise ASMRuntimeError(f"{name} expects at least {builtin.min_args} arguments", rewrite_rule=name, location=location)
        if builtin.max_args is not None and supplied > builtin.max_args:
            raise ASMRuntimeError(f"{name} expects at most {builtin.max_args} arguments", rewrite_rule=name, location=location)
        return builtin.impl(interpreter, args, arg_nodes, env, location)

    # Helpers
    def _expect_int(self, value: Value, rule: str, location: SourceLocation) -> int:
        if value.type != TYPE_INT:
            raise ASMRuntimeError(f"{rule} expects integer arguments", location=location, rewrite_rule=rule)
        assert isinstance(value.value, int)
        return value.value

    def _expect_str(self, value: Value, rule: str, location: SourceLocation) -> str:
        if value.type != TYPE_STR:
            raise ASMRuntimeError(f"{rule} expects string arguments", location=location, rewrite_rule=rule)
        assert isinstance(value.value, str)
        return value.value

    def _expect_tns(self, value: Value, rule: str, location: SourceLocation) -> Tensor:
        if value.type != TYPE_TNS:
            raise ASMRuntimeError(f"{rule} expects tensor arguments", location=location, rewrite_rule=rule)
        assert isinstance(value.value, Tensor)
        return value.value

    def _normalize_coding(self, coding_raw: str, rule: str, location: SourceLocation) -> str:
        tag = coding_raw.strip().lower().replace("_", "-")
        compact = tag.replace("-", "").replace(" ", "")
        mapping = {
            "utf8": "utf-8",
            "utf8bom": "utf-8-bom",
            "utf8sig": "utf-8-bom",
            "utf": "utf-8",
            "utf-8": "utf-8",
            "utf-8bom": "utf-8-bom",
            "utf-8sig": "utf-8-bom",
            "utf16le": "utf-16-le",
            "utf16be": "utf-16-be",
            "utf-16le": "utf-16-le",
            "utf-16be": "utf-16-be",
            "binary": "binary",
            "bin": "binary",
            "hex": "hex",
            "hexadecimal": "hex",
            "ansi": "ansi",
        }
        normalized = mapping.get(compact)
        if normalized is None:
            raise ASMRuntimeError(
                f"Unsupported coding '{coding_raw}'",
                location=location,
                rewrite_rule=rule,
            )
        return normalized

    def _extract_powershell_body(self, cmd: str) -> Optional[str]:
        """Best-effort extraction of the script text passed to -Command.

        This keeps the heavy script contents out of the CreateProcess command
        line by allowing us to emit it into a temporary .ps1 file instead.
        """
        lower = cmd.lower()
        idx = lower.find("-command")
        if idx == -1:
            return None
        tail = cmd[idx + len("-command") :].lstrip()
        if not tail:
            return None
        if tail[0] in ('"', "'"):
            quote = tail[0]
            tail = tail[1:]
            end = tail.find(quote)
            if end == -1:
                return None
            return tail[:end]
        # Fallback: take the next token
        parts = tail.split(None, 1)
        return parts[0] if parts else None

    def _as_bool_value(self, value: Value) -> int:
        if value.type == TYPE_INT:
            return 0 if value.value == 0 else 1
        if value.type == TYPE_STR:
            return 0 if value.value == "" else 1
        if value.type == TYPE_TNS:
            assert isinstance(value.value, Tensor)
            return 1 if any(self._as_bool_value(item) for item in value.value.data.flat) else 0
        return 0

    def _condition_from_value(self, value: Value) -> int:
        if value.type == TYPE_INT:
            return value.value  # type: ignore[return-value]
        if value.type == TYPE_STR:
            text = value.value  # type: ignore[assignment]
            if text == "":
                return 0
            if set(text).issubset({"0", "1"}):
                return int(text, 2)
            return 1
        if value.type == TYPE_TNS:
            return self._as_bool_value(value)
        return 0

    def _safe_div(self, a: int, b: int) -> int:
        if b == 0:
            raise ASMRuntimeError("Division by zero", rewrite_rule="DIV")
        return a // b

    def _safe_cdiv(self, a: int, b: int) -> int:
        if b == 0:
            raise ASMRuntimeError("Division by zero", rewrite_rule="CDIV")
        q = a // b
        if a % b == 0:
            return q
        return q + 1

    def _safe_mod(self, a: int, b: int) -> int:
        if b == 0:
            raise ASMRuntimeError("Division by zero", rewrite_rule="MOD")
        return a % abs(b)

    def _safe_pow(self, a: int, b: int) -> int:
        if b < 0:
            raise ASMRuntimeError("Negative exponent not supported", rewrite_rule="POW")
        return pow(a, b)

    def _shift_left(self, value: int, amount: int) -> int:
        if amount < 0:
            raise ASMRuntimeError("SHL amount must be non-negative", rewrite_rule="SHL")
        return value << amount

    def _shift_right(self, value: int, amount: int) -> int:
        if amount < 0:
            raise ASMRuntimeError("SHR amount must be non-negative", rewrite_rule="SHR")
        return value >> amount

    def _lcm(self, a: int, b: int) -> int:
        if a == 0 or b == 0:
            return 0
        return abs(a * b) // math.gcd(a, b)

    # Variadic helpers
    def _sum(self, values: List[Value], location: SourceLocation) -> Value:
        ints = [self._expect_int(v, "SUM", location) for v in values]
        return Value(TYPE_INT, sum(ints))

    def _prod(self, values: List[Value], location: SourceLocation) -> Value:
        ints = [self._expect_int(v, "PROD", location) for v in values]
        result = 1
        for val in ints:
            result *= val
        return Value(TYPE_INT, result)

    def _max(self, values: List[Value], location: SourceLocation) -> Value:
        if not values:
            raise ASMRuntimeError("MAX requires at least one argument", rewrite_rule="MAX")
        first_type = values[0].type
        if first_type == TYPE_TNS:
            raise ASMRuntimeError("MAX cannot operate on tensors", rewrite_rule="MAX", location=location)
        if any(v.type != first_type for v in values):
            raise ASMRuntimeError("MAX cannot mix integers and strings", rewrite_rule="MAX", location=location)
        if first_type == TYPE_INT:
            ints = [self._expect_int(v, "MAX", location) for v in values]
            return Value(TYPE_INT, max(ints))
        strs = [self._expect_str(v, "MAX", location) for v in values]
        longest = max(strs, key=len)
        return Value(TYPE_STR, longest)

    def _min(self, values: List[Value], location: SourceLocation) -> Value:
        if not values:
            raise ASMRuntimeError("MIN requires at least one argument", rewrite_rule="MIN")
        first_type = values[0].type
        if first_type == TYPE_TNS:
            raise ASMRuntimeError("MIN cannot operate on tensors", rewrite_rule="MIN", location=location)
        if any(v.type != first_type for v in values):
            raise ASMRuntimeError("MIN cannot mix integers and strings", rewrite_rule="MIN", location=location)
        if first_type == TYPE_INT:
            ints = [self._expect_int(v, "MIN", location) for v in values]
            return Value(TYPE_INT, min(ints))
        strs = [self._expect_str(v, "MIN", location) for v in values]
        shortest = min(strs, key=len)
        return Value(TYPE_STR, shortest)

    def _any(self, values: List[Value], _: SourceLocation) -> Value:
        return Value(TYPE_INT, 1 if any(self._as_bool_value(v) for v in values) else 0)

    def _all(self, values: List[Value], _: SourceLocation) -> Value:
        return Value(TYPE_INT, 1 if all(self._as_bool_value(v) for v in values) else 0)

    def _len(self, values: List[Value], _: SourceLocation) -> Value:
        for v in values:
            if v.type not in (TYPE_INT, TYPE_STR):
                raise ASMRuntimeError("LEN accepts only INT or STR arguments", rewrite_rule="LEN")
        return Value(TYPE_INT, len(values))

    def _slen(
        self,
        _: "Interpreter",
        args: List[Value],
        __: List[Expression],
        ___: Environment,
        location: SourceLocation,
    ) -> Value:
        text = self._expect_str(args[0], "SLEN", location)
        return Value(TYPE_INT, len(text))

    def _ilen(
        self,
        _: "Interpreter",
        args: List[Value],
        __: List[Expression],
        ___: Environment,
        location: SourceLocation,
    ) -> Value:
        number = self._expect_int(args[0], "ILEN", location)
        magnitude = abs(number)
        length = 1 if magnitude == 0 else magnitude.bit_length()
        return Value(TYPE_INT, length)

    def _join(self, values: List[Value], location: SourceLocation) -> Value:
        if not values:
            raise ASMRuntimeError("JOIN requires at least one argument", rewrite_rule="JOIN")
        first_type = values[0].type
        if first_type == TYPE_TNS:
            raise ASMRuntimeError("JOIN cannot operate on tensors", rewrite_rule="JOIN", location=location)
        if any(v.type != first_type for v in values):
            raise ASMRuntimeError("JOIN cannot mix integers and strings", rewrite_rule="JOIN", location=location)
        if first_type == TYPE_STR:
            parts = [self._expect_str(v, "JOIN", location) for v in values]
            return Value(TYPE_STR, "".join(parts))

        ints = [self._expect_int(v, "JOIN", location) for v in values]
        if any(val < 0 for val in ints):
            if not all(val < 0 for val in ints):
                raise ASMRuntimeError("JOIN arguments must not mix positive and negative values", rewrite_rule="JOIN")
            abs_vals = [abs(v) for v in ints]
            bits = "".join("0" if v == 0 else format(v, "b") for v in abs_vals)
            return Value(TYPE_INT, -int(bits or "0", 2))
        bits = "".join("0" if v == 0 else format(v, "b") for v in ints)
        return Value(TYPE_INT, int(bits or "0", 2))

    def _split(
        self,
        _: "Interpreter",
        args: List[Value],
        __: List[Expression],
        ___: Environment,
        location: SourceLocation,
    ) -> Value:
        text = self._expect_str(args[0], "SPLIT", location)
        delimiter = " " if len(args) == 1 else self._expect_str(args[1], "SPLIT", location)
        if delimiter == "":
            raise ASMRuntimeError("SPLIT delimiter must not be empty", location=location, rewrite_rule="SPLIT")

        parts = text.split(delimiter)
        data = np.array([Value(TYPE_STR, part) for part in parts], dtype=object)
        return Value(TYPE_TNS, Tensor(shape=[len(parts)], data=data))

    # Boolean-like operators treating strings via emptiness
    def _and(self, _: "Interpreter", args: List[Value], __: List[Expression], ___: Environment, location: SourceLocation) -> Value:
        a, b = args
        return Value(TYPE_INT, 1 if (self._as_bool_value(a) and self._as_bool_value(b)) else 0)

    def _or(self, _: "Interpreter", args: List[Value], __: List[Expression], ___: Environment, location: SourceLocation) -> Value:
        a, b = args
        return Value(TYPE_INT, 1 if (self._as_bool_value(a) or self._as_bool_value(b)) else 0)

    def _xor(self, _: "Interpreter", args: List[Value], __: List[Expression], ___: Environment, location: SourceLocation) -> Value:
        a, b = args
        return Value(TYPE_INT, 1 if (self._as_bool_value(a) ^ self._as_bool_value(b)) else 0)

    def _not(self, _: "Interpreter", args: List[Value], __: List[Expression], ___: Environment, ___loc: SourceLocation) -> Value:
        return Value(TYPE_INT, 1 if self._as_bool_value(args[0]) == 0 else 0)

    def _bool(self, _: "Interpreter", args: List[Value], __: List[Expression], ___: Environment, __loc: SourceLocation) -> Value:
        # BOOL(ANY: item):INT -> truthiness of item (INT: nonzero, STR: non-empty, TNS: any true element)
        return Value(TYPE_INT, 1 if self._as_bool_value(args[0]) != 0 else 0)

    def _eq(self, interpreter: "Interpreter", args: List[Value], __: List[Expression], ___: Environment, ___loc: SourceLocation) -> Value:
        a, b = args
        if a.type != b.type:
            return Value(TYPE_INT, 0)
        if a.type == TYPE_TNS:
            assert isinstance(a.value, Tensor)
            assert isinstance(b.value, Tensor)
            return Value(TYPE_INT, 1 if interpreter._tensor_equal(a.value, b.value) else 0)
        return Value(TYPE_INT, 1 if a.value == b.value else 0)

    def _in(self, interpreter: "Interpreter", args: List[Value], __: List[Expression], ___: Environment, location: SourceLocation) -> Value:
        # IN(ANY: value, TNS: tensor):INT -> 1 if value is contained anywhere in tensor, else 0
        if len(args) != 2:
            raise ASMRuntimeError("IN requires two arguments", location=location, rewrite_rule="IN")
        needle, haystack = args
        if haystack.type != TYPE_TNS:
            raise ASMRuntimeError("IN requires a tensor as second argument", location=location, rewrite_rule="IN")
        assert isinstance(haystack.value, Tensor)
        for item in haystack.value.data.flat:
            if interpreter._values_equal(needle, item):
                return Value(TYPE_INT, 1)
        return Value(TYPE_INT, 0)

    def _slice(self, _: "Interpreter", args: List[Value], __: List[Expression], ___: Environment, location: SourceLocation) -> Value:
        target, hi_val, lo_val = args
        hi = self._expect_int(hi_val, "SLICE", location)
        lo = self._expect_int(lo_val, "SLICE", location)
        if hi < lo:
            raise ASMRuntimeError("SLICE: hi must be >= lo", rewrite_rule="SLICE", location=location)
        if hi < 0 or lo < 0:
            raise ASMRuntimeError("SLICE: indices must be non-negative", rewrite_rule="SLICE", location=location)
        if target.type == TYPE_INT:
            width = hi - lo + 1
            if width <= 0:
                return Value(TYPE_INT, 0)
            mask = (1 << (hi + 1)) - 1
            result = (self._expect_int(target, "SLICE", location) & mask) >> lo
            return Value(TYPE_INT, result)
        if target.type != TYPE_STR:
            raise ASMRuntimeError("SLICE target must be int or string", rewrite_rule="SLICE", location=location)
        text = self._expect_str(target, "SLICE", location)
        length = len(text)
        start = length - 1 - hi
        end = length - 1 - lo
        if start < 0 or end < 0 or start > end:
            raise ASMRuntimeError("SLICE indices out of range for string", rewrite_rule="SLICE", location=location)
        return Value(TYPE_STR, text[start : end + 1])

    def _int_op(self, _: "Interpreter", args: List[Value], __: List[Expression], ___: Environment, location: SourceLocation) -> Value:
        val = args[0]
        if val.type == TYPE_INT:
            return val
        text = self._expect_str(val, "INT", location)
        if text == "":
            return Value(TYPE_INT, 0)
        if set(text).issubset({"0", "1"}):
            return Value(TYPE_INT, int(text, 2))
        return Value(TYPE_INT, 1)

    def _str_op(self, _: "Interpreter", args: List[Value], __: List[Expression], ___: Environment, location: SourceLocation) -> Value:
        val = args[0]
        if val.type == TYPE_STR:
            return val
        number = self._expect_int(val, "STR", location)
        if number < 0:
            return Value(TYPE_STR, "-" + format(-number, "b"))
        return Value(TYPE_STR, format(number, "b"))

    def _upper(
        self,
        interpreter: "Interpreter",
        args: List[Value],
        __: List[Expression],
        ___: Environment,
        location: SourceLocation,
    ) -> Value:
        s = self._expect_str(args[0], "UPPER", location)
        # Convert ASCII letters to upper-case; other bytes are unchanged
        return Value(TYPE_STR, s.upper())

    def _lower(
        self,
        interpreter: "Interpreter",
        args: List[Value],
        __: List[Expression],
        ___: Environment,
        location: SourceLocation,
    ) -> Value:
        s = self._expect_str(args[0], "LOWER", location)
        # Convert ASCII letters to lower-case; other bytes are unchanged
        return Value(TYPE_STR, s.lower())

    def _strip(
        self,
        interpreter: "Interpreter",
        args: List[Value],
        __: List[Expression],
        ___: Environment,
        location: SourceLocation,
    ) -> Value:
        # STRIP(STR: string, STR: remove):STR -> remove all occurrences of `remove` from `string`
        s = self._expect_str(args[0], "STRIP", location)
        rem = self._expect_str(args[1], "STRIP", location)
        if rem == "":
            raise ASMRuntimeError("STRIP: remove substring must not be empty", location=location, rewrite_rule="STRIP")
        return Value(TYPE_STR, s.replace(rem, ""))

    def _replace(
        self,
        interpreter: "Interpreter",
        args: List[Value],
        __: List[Expression],
        ___: Environment,
        location: SourceLocation,
    ) -> Value:
        # REPLACE(STR: string, STR: a, STR: b):STR -> replace all occurrences of `a` in `string` with `b`
        s = self._expect_str(args[0], "REPLACE", location)
        a = self._expect_str(args[1], "REPLACE", location)
        b = self._expect_str(args[2], "REPLACE", location)
        if a == "":
            raise ASMRuntimeError("REPLACE: substring must not be empty", location=location, rewrite_rule="REPLACE")
        return Value(TYPE_STR, s.replace(a, b))

    def _safe_log(self, value: int) -> int:
        if value <= 0:
            raise ASMRuntimeError("LOG argument must be > 0", rewrite_rule="LOG")
        return value.bit_length() - 1

    def _safe_clog(self, value: int) -> int:
        if value <= 0:
            raise ASMRuntimeError("CLOG argument must be > 0", rewrite_rule="CLOG")
        if value & (value - 1) == 0:
            return value.bit_length() - 1
        return value.bit_length()

    def _main(
        self,
        interpreter: "Interpreter",
        _: List[Value],
        __: List[Expression],
        ___: Environment,
        location: SourceLocation,
    ) -> Value:
        root = interpreter.entry_filename
        if root == "<string>":
            return Value(TYPE_INT, 1 if location.file == "<string>" else 0)
        return Value(TYPE_INT, 1 if os.path.abspath(location.file) == root else 0)

    def _os(
        self,
        interpreter: "Interpreter",
        args: List[Value],
        __: List[Expression],
        ___: Environment,
        __loc: SourceLocation,
    ) -> Value:
        # Return a short lowercase host OS family string as STR.
        plat = sys.platform.lower()
        if plat.startswith("win") or plat.startswith("cygwin"):
            fam = "win"
        elif plat.startswith("linux"):
            fam = "linux"
        elif plat.startswith("darwin"):
            fam = "macos"
        elif plat.startswith("aix") or plat.startswith("freebsd") or plat.startswith("openbsd"):
            fam = "unix"
        else:
            # Fallback to platform.system() for additional hints
            p = platform.system().lower()
            if "windows" in p:
                fam = "win"
            elif "linux" in p:
                fam = "linux"
            elif "darwin" in p or "mac" in p:
                fam = "macos"
            elif p:
                fam = p
            else:
                fam = "unix"
        return Value(TYPE_STR, fam)

    def _import(
        self,
        interpreter: "Interpreter",
        _: List[Value],
        arg_nodes: List[Expression],
        env: Environment,
        location: SourceLocation,
    ) -> Value:
        if len(arg_nodes) != 1 or not isinstance(arg_nodes[0], Identifier):
            raise ASMRuntimeError("IMPORT expects module name identifier", location=location, rewrite_rule="IMPORT")

        module_name = arg_nodes[0].name
        # If module was already imported earlier in this interpreter instance,
        # reuse the same Environment and function objects so all importers
        # observe the same namespace/instance.
        if module_name in interpreter.module_cache:
            cached_env = interpreter.module_cache[module_name]
            # Ensure module functions are registered in the interpreter.functions map
            for fn in interpreter.module_functions.get(module_name, []):
                if fn.name not in interpreter.functions:
                    interpreter.functions[fn.name] = fn

            for k, v in cached_env.values.items():
                dotted = f"{module_name}.{k}"
                env.set(dotted, v, declared_type=v.type)
            return Value(TYPE_INT, 0)

        base_dir = os.getcwd() if location.file == "<string>" else os.path.dirname(os.path.abspath(location.file))
        module_path = os.path.join(base_dir, f"{module_name}.asmln")

        try:
            with open(module_path, "r", encoding="utf-8") as handle:
                source_text = handle.read()
        except OSError as exc:
            interpreter_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
            lib_module_path = os.path.join(interpreter_dir, "lib", f"{module_name}.asmln")
            try:
                with open(lib_module_path, "r", encoding="utf-8") as handle:
                    source_text = handle.read()
                    module_path = lib_module_path
            except OSError:
                raise ASMRuntimeError(f"Failed to import '{module_name}': {exc}", location=location, rewrite_rule="IMPORT")

        lexer = Lexer(source_text, module_path)
        tokens = lexer.tokenize()
        parser = Parser(tokens, module_path, source_text.splitlines())
        program = parser.parse()

        module_env = Environment()
        prev_functions = dict(interpreter.functions)
        try:
            interpreter._execute_block(program.statements, module_env)
        except Exception:
            interpreter.functions = prev_functions
            raise

        # Collect functions that were added by executing the module
        new_funcs = {n: f for n, f in interpreter.functions.items() if n not in prev_functions}
        registered_functions: List[Function] = []
        for name, fn in new_funcs.items():
            dotted_name = f"{module_name}.{name}"
            if "." in name:
                created = Function(
                    name=dotted_name,
                    params=fn.params,
                    return_type=fn.return_type,
                    body=fn.body,
                    closure=fn.closure,
                )
                interpreter.functions[dotted_name] = created
                registered_functions.append(created)
            else:
                # move unqualified function into module-qualified function
                interpreter.functions.pop(name, None)
                created = Function(
                    name=dotted_name,
                    params=fn.params,
                    return_type=fn.return_type,
                    body=fn.body,
                    closure=module_env,
                )
                interpreter.functions[dotted_name] = created
                registered_functions.append(created)

        # Store module env and functions into the interpreter cache for reuse
        interpreter.module_cache[module_name] = module_env
        interpreter.module_functions[module_name] = registered_functions

        # Export top-level bindings from the module under the dotted namespace
        for k, v in module_env.values.items():
            dotted = f"{module_name}.{k}"
            env.set(dotted, v, declared_type=v.type)
        return Value(TYPE_INT, 0)

    def _run(
        self,
        interpreter: "Interpreter",
        args: List[Value],
        __: List[Expression],
        env: Environment,
        location: SourceLocation,
    ) -> Value:
        # RUN(source): execute the provided source code string within the
        # current environment (mutating `env` and `interpreter.functions`).
        source_text = self._expect_str(args[0], "RUN", location)
        run_filename = location.file if location and location.file else "<run>"

        lexer = Lexer(source_text, run_filename)
        tokens = lexer.tokenize()
        parser = Parser(tokens, run_filename, source_text.splitlines())
        program = parser.parse()

        # Execute parsed statements in the caller's environment so that
        # assignments and function definitions are visible to the caller.
        # RUN is explicitly allowed to produce console output even when
        # the interpreter is currently shushed; temporarily disable
        # shushing while executing the supplied source.
        old_shush = interpreter.shushed
        try:
            interpreter.shushed = False
            interpreter._execute_block(program.statements, env)
        finally:
            interpreter.shushed = old_shush
        return Value(TYPE_INT, 0)

    def _input(
        self,
        interpreter: "Interpreter",
        args: List[Value],
        __: List[Expression],
        ___: Environment,
        ____: SourceLocation,
    ) -> Value:
        # Accept an optional prompt string. If provided, print it before
        # requesting input so callers can display an inline prompt.
        prompt: Optional[str] = None
        if args:
            prompt = self._expect_str(args[0], "INPUT", ____)
            # Emit the prompt to the output sink so callers observing output
            # (for example, a REPL) see it. This may include a trailing
            # newline depending on the configured sink.
            interpreter.output_sink(prompt)

        text = interpreter.input_provider()
        # Record prompt when present to make I/O replay deterministic.
        record: Dict[str, Any] = {"event": "INPUT", "text": text}
        if prompt is not None:
            record["prompt"] = prompt
        interpreter.io_log.append(record)
        return Value(TYPE_STR, text)

    def _print(
        self,
        interpreter: "Interpreter",
        args: List[Value],
        __: List[Expression],
        ___: Environment,
        ____: SourceLocation,
    ) -> Value:
        loc = ____
        rendered: List[str] = []
        for arg in args:
            if arg.type == TYPE_INT:
                number = self._expect_int(arg, "PRINT", ____)
                rendered.append(("-" + format(-number, "b")) if number < 0 else format(number, "b"))
            elif arg.type == TYPE_STR:
                rendered.append(arg.value)  # type: ignore[arg-type]
            else:
                raise ASMRuntimeError("PRINT accepts INT or STR arguments", location=loc, rewrite_rule="PRINT")
        text = "".join(rendered)
        # Forward to configured output sink only when not shushed.
        if not interpreter.shushed:
            interpreter.output_sink(text)
        interpreter.io_log.append({"event": "PRINT", "values": [arg.value for arg in args]})
        return Value(TYPE_INT, 0)

    def _argv(
        self,
        interpreter: "Interpreter",
        args: List[Value],
        __: List[Expression],
        ___: Environment,
        ____: SourceLocation,
    ) -> Value:
        # Return the process argument vector as a 1-D TNS of STR values.
        entries: List[str] = [str(s) for s in sys.argv]
        data = np.array([Value(TYPE_STR, s) for s in entries], dtype=object)
        return Value(TYPE_TNS, Tensor(shape=[len(data)], data=data))

    def _assert(
        self,
        interpreter: "Interpreter",
        args: List[Value],
        __: List[Expression],
        ___: Environment,
        location: SourceLocation,
    ) -> Value:
        cond = self._condition_from_value(args[0]) if args else 0
        if cond == 0:
            raise ASMRuntimeError("Assertion failed", location=location, rewrite_rule="ASSERT")
        return Value(TYPE_INT, 1)

    def _delete(
        self,
        interpreter: "Interpreter",
        args: List[Value],
        arg_nodes: List[Expression],
        env: Environment,
        location: SourceLocation,
    ) -> Value:
        if not arg_nodes or not isinstance(arg_nodes[0], Identifier):
            raise ASMRuntimeError("DEL expects identifier", location=location, rewrite_rule="DEL")
        name = arg_nodes[0].name
        try:
            env.delete(name)
        except ASMRuntimeError as err:
            err.location = location
            raise
        return Value(TYPE_INT, 0)

    def _freeze(
        self,
        interpreter: "Interpreter",
        args: List[Value],
        arg_nodes: List[Expression],
        env: Environment,
        location: SourceLocation,
    ) -> Value:
        if not arg_nodes or not isinstance(arg_nodes[0], Identifier):
            raise ASMRuntimeError("FREEZE expects identifier", location=location, rewrite_rule="FREEZE")
        name = arg_nodes[0].name
        try:
            env.freeze(name)
        except ASMRuntimeError as err:
            err.location = location
            raise
        return Value(TYPE_INT, 0)

    def _thaw(
        self,
        interpreter: "Interpreter",
        args: List[Value],
        arg_nodes: List[Expression],
        env: Environment,
        location: SourceLocation,
    ) -> Value:
        if not arg_nodes or not isinstance(arg_nodes[0], Identifier):
            raise ASMRuntimeError("THAW expects identifier", location=location, rewrite_rule="THAW")
        name = arg_nodes[0].name
        try:
            env.thaw(name)
        except ASMRuntimeError as err:
            err.location = location
            raise
        return Value(TYPE_INT, 0)

    def _permafreeze(
        self,
        interpreter: "Interpreter",
        args: List[Value],
        arg_nodes: List[Expression],
        env: Environment,
        location: SourceLocation,
    ) -> Value:
        if not arg_nodes or not isinstance(arg_nodes[0], Identifier):
            raise ASMRuntimeError("PERMAFREEZE expects identifier", location=location, rewrite_rule="PERMAFREEZE")
        name = arg_nodes[0].name
        try:
            env.permafreeze(name)
        except ASMRuntimeError as err:
            err.location = location
            raise
        return Value(TYPE_INT, 0)

    def _export(
        self,
        interpreter: "Interpreter",
        args: List[Value],
        arg_nodes: List[Expression],
        env: Environment,
        location: SourceLocation,
    ) -> Value:
        # Expect two identifier nodes: the symbol to export and the module name.
        if not arg_nodes or len(arg_nodes) < 2:
            raise ASMRuntimeError("EXPORT expects two identifier arguments", location=location, rewrite_rule="EXPORT")
        if not isinstance(arg_nodes[0], Identifier):
            raise ASMRuntimeError("EXPORT expects an identifier as the first argument", location=location, rewrite_rule="EXPORT")
        if not isinstance(arg_nodes[1], Identifier):
            raise ASMRuntimeError("EXPORT expects an identifier as the second argument", location=location, rewrite_rule="EXPORT")

        symbol_name = arg_nodes[0].name
        module_name = arg_nodes[1].name

        # Ensure the symbol exists in the caller's environment
        if not env.has(symbol_name):
            raise ASMRuntimeError(f"EXPORT: undefined symbol '{symbol_name}'", location=location, rewrite_rule="EXPORT")
        try:
            value = env.get(symbol_name)
        except ASMRuntimeError as err:
            err.location = location
            raise

        # Detect whether the module has been imported into this environment.
        # We consider a module imported if there exists any dotted binding
        # beginning with 'module_name.' in the current environment or if
        # any function name is registered under that module prefix.
        prefix = f"{module_name}."
        module_present = any(k.startswith(prefix) for k in env.values.keys()) or any(k.startswith(prefix) for k in interpreter.functions.keys())
        if not module_present:
            raise ASMRuntimeError(f"Module '{module_name}' not imported", location=location, rewrite_rule="EXPORT")

        dotted = f"{module_name}.{symbol_name}"
        # Create or update the dotted binding in the caller's environment.
        env.set(dotted, value, declared_type=value.type)
        return Value(TYPE_INT, 0)

    def _exist(
        self,
        interpreter: "Interpreter",
        args: List[Value],
        arg_nodes: List[Expression],
        env: Environment,
        location: SourceLocation,
    ) -> Value:
        if not arg_nodes or not isinstance(arg_nodes[0], Identifier):
            raise ASMRuntimeError("EXIST requires an identifier argument", location=location, rewrite_rule="EXIST")
        name = arg_nodes[0].name
        return Value(TYPE_INT, 1 if env.has(name) else 0)

    def _isint(
        self,
        interpreter: "Interpreter",
        args: List[Value],
        arg_nodes: List[Expression],
        env: Environment,
        location: SourceLocation,
    ) -> Value:
        # Expect an identifier node so we examine the symbol's binding and type
        if not arg_nodes or not isinstance(arg_nodes[0], Identifier):
            raise ASMRuntimeError("ISINT requires an identifier argument", location=location, rewrite_rule="ISINT")
        name = arg_nodes[0].name
        if not env.has(name):
            return Value(TYPE_INT, 0)
        try:
            val = env.get(name)
        except ASMRuntimeError as err:
            err.location = location
            raise
        return Value(TYPE_INT, 1 if val.type == TYPE_INT else 0)

    def _isstr(
        self,
        interpreter: "Interpreter",
        args: List[Value],
        arg_nodes: List[Expression],
        env: Environment,
        location: SourceLocation,
    ) -> Value:
        # Expect an identifier node so we examine the symbol's binding and type
        if not arg_nodes or not isinstance(arg_nodes[0], Identifier):
            raise ASMRuntimeError("ISSTR requires an identifier argument", location=location, rewrite_rule="ISSTR")
        name = arg_nodes[0].name
        if not env.has(name):
            return Value(TYPE_INT, 0)
        try:
            val = env.get(name)
        except ASMRuntimeError as err:
            err.location = location
            raise
        return Value(TYPE_INT, 1 if val.type == TYPE_STR else 0)

    def _istns(
        self,
        interpreter: "Interpreter",
        args: List[Value],
        arg_nodes: List[Expression],
        env: Environment,
        location: SourceLocation,
    ) -> Value:
        if not arg_nodes or not isinstance(arg_nodes[0], Identifier):
            raise ASMRuntimeError("ISTNS requires an identifier argument", location=location, rewrite_rule="ISTNS")
        name = arg_nodes[0].name
        if not env.has(name):
            return Value(TYPE_INT, 0)
        try:
            val = env.get(name)
        except ASMRuntimeError as err:
            err.location = location
            raise
        return Value(TYPE_INT, 1 if val.type == TYPE_TNS else 0)

    def _frozen(
        self,
        interpreter: "Interpreter",
        args: List[Value],
        arg_nodes: List[Expression],
        env: Environment,
        location: SourceLocation,
    ) -> Value:
        # Expect an identifier node so we examine whether the symbol is frozen
        if not arg_nodes or not isinstance(arg_nodes[0], Identifier):
            raise ASMRuntimeError("FROZEN requires an identifier argument", location=location, rewrite_rule="FROZEN")
        name = arg_nodes[0].name
        env_found = env._find_env(name)
        if env_found is None:
            return Value(TYPE_INT, 0)
        # Permafrozen takes precedence: return -1 for permanently-frozen
        if name in env_found.permafrozen:
            return Value(TYPE_INT, -1)
        return Value(TYPE_INT, 1 if name in env_found.frozen else 0)

    def _permafrozen(
        self,
        interpreter: "Interpreter",
        args: List[Value],
        arg_nodes: List[Expression],
        env: Environment,
        location: SourceLocation,
    ) -> Value:
        # Expect an identifier node so we examine whether the symbol is perma-frozen
        if not arg_nodes or not isinstance(arg_nodes[0], Identifier):
            raise ASMRuntimeError("PERMAFROZEN requires an identifier argument", location=location, rewrite_rule="PERMAFROZEN")
        name = arg_nodes[0].name
        env_found = env._find_env(name)
        if env_found is None:
            return Value(TYPE_INT, 0)
        return Value(TYPE_INT, 1 if name in env_found.permafrozen else 0)

    def _readfile(
        self,
        interpreter: "Interpreter",
        args: List[Value],
        __: List[Expression],
        ___: Environment,
        location: SourceLocation,
    ) -> Value:
        path = self._expect_str(args[0], "READFILE", location)
        coding_val = args[1] if len(args) > 1 else Value(TYPE_STR, "UTF-8")
        coding_text = self._expect_str(coding_val, "READFILE", location)
        coding = self._normalize_coding(coding_text, "READFILE", location)

        if coding in {"binary", "hex"}:
            try:
                with open(path, "rb") as handle:
                    raw = handle.read()
            except OSError as exc:
                raise ASMRuntimeError(f"Failed to read '{path}': {exc}", location=location, rewrite_rule="READFILE")

            if coding == "binary":
                text = "".join(f"{byte:08b}" for byte in raw)
            else:
                text = raw.hex()
            return Value(TYPE_STR, text)

        encoding = "utf-8-sig" if coding in {"utf-8", "utf-8-bom"} else coding
        if coding == "ansi":
            encoding = "cp1252" if os.name == "nt" else "latin-1"
        try:
            with open(path, "r", encoding=encoding, errors="replace") as handle:
                data = handle.read()
        except OSError as exc:
            raise ASMRuntimeError(f"Failed to read '{path}': {exc}", location=location, rewrite_rule="READFILE")
        return Value(TYPE_STR, data)

    def _writefile(
        self,
        interpreter: "Interpreter",
        args: List[Value],
        __: List[Expression],
        ___: Environment,
        location: SourceLocation,
    ) -> Value:
        blob = self._expect_str(args[0], "WRITEFILE", location)
        path = self._expect_str(args[1], "WRITEFILE", location)
        coding_val = args[2] if len(args) > 2 else Value(TYPE_STR, "UTF-8")
        coding_text = self._expect_str(coding_val, "WRITEFILE", location)
        coding = self._normalize_coding(coding_text, "WRITEFILE", location)

        try:
            if coding == "binary":
                cleaned = "".join(ch for ch in blob if not ch.isspace())
                if any(ch not in "01" for ch in cleaned):
                    raise ASMRuntimeError("WRITEFILE(binary) expects only 0/1 characters", location=location, rewrite_rule="WRITEFILE")
                if len(cleaned) % 8 != 0:
                    raise ASMRuntimeError("WRITEFILE(binary) requires bitstrings in multiples of 8 bits", location=location, rewrite_rule="WRITEFILE")
                out_bytes = bytes(int(cleaned[i : i + 8], 2) for i in range(0, len(cleaned), 8))
                with open(path, "wb") as handle:
                    handle.write(out_bytes)
            elif coding == "hex":
                try:
                    out_bytes = bytes.fromhex(blob)
                except ValueError as exc:
                    raise ASMRuntimeError(f"WRITEFILE(hex) could not parse hex: {exc}", location=location, rewrite_rule="WRITEFILE")
                with open(path, "wb") as handle:
                    handle.write(out_bytes)
            else:
                encoding = "utf-8" if coding == "utf-8" else coding
                if coding == "utf-8-bom":
                    encoding = "utf-8-sig"
                if coding == "ansi":
                    encoding = "cp1252" if os.name == "nt" else "latin-1"
                with open(path, "w", encoding=encoding) as handle:
                    handle.write(blob)
        except OSError:
            return Value(TYPE_INT, 0)
        return Value(TYPE_INT, 1)

    def _bytes(
        self,
        interpreter: "Interpreter",
        args: List[Value],
        __: List[Expression],
        ___: Environment,
        location: SourceLocation,
    ) -> Value:
        number = self._expect_int(args[0], "BYTES", location)
        if number < 0:
            raise ASMRuntimeError("BYTES expects a non-negative integer", location=location, rewrite_rule="BYTES")
        if number == 0:
            data = np.array([Value(TYPE_INT, 0)], dtype=object)
            return Value(TYPE_TNS, Tensor(shape=[1], data=data))

        octets: List[int] = []
        temp = number
        while temp > 0:
            octets.append(temp & 0xFF)
            temp >>= 8
        octets.reverse()

        data = np.array([Value(TYPE_INT, b) for b in octets], dtype=object)
        return Value(TYPE_TNS, Tensor(shape=[len(octets)], data=data))

    def _existfile(
        self,
        interpreter: "Interpreter",
        args: List[Value],
        __: List[Expression],
        ___: Environment,
        location: SourceLocation,
    ) -> Value:
        path = self._expect_str(args[0], "EXISTFILE", location)
        return Value(TYPE_INT, 1 if os.path.exists(path) else 0)

    # Tensor helpers
    def _shape_from_tensor(self, tensor: Tensor, rule: str, location: SourceLocation) -> List[int]:
        if len(tensor.shape) != 1:
            raise ASMRuntimeError(f"{rule} shape must be a 1D tensor", location=location, rewrite_rule=rule)
        dims: List[int] = []
        for entry in tensor.data.flat:
            dim = self._expect_int(entry, rule, location)
            if dim <= 0:
                raise ASMRuntimeError("Tensor dimensions must be positive", location=location, rewrite_rule=rule)
            dims.append(dim)
        if not dims:
            raise ASMRuntimeError("Tensor shape must have at least one dimension", location=location, rewrite_rule=rule)
        return dims

    def _map_tensor_int_binary(self, x: Tensor, y: Tensor, rule: str, location: SourceLocation, op: Callable[[int, int], int]) -> Tensor:
        if x.shape != y.shape:
            raise ASMRuntimeError(f"{rule} requires tensors with identical shapes", location=location, rewrite_rule=rule)
        data = np.array(
            [
                Value(TYPE_INT, op(self._expect_int(a, rule, location), self._expect_int(b, rule, location)))
                for a, b in zip(x.data.flat, y.data.flat)
            ],
            dtype=object,
        )
        return Tensor(shape=list(x.shape), data=data)

    def _map_tensor_int_scalar(self, tensor: Tensor, scalar: int, rule: str, location: SourceLocation, op: Callable[[int, int], int]) -> Tensor:
        data = np.array(
            [Value(TYPE_INT, op(self._expect_int(entry, rule, location), scalar)) for entry in tensor.data.flat],
            dtype=object,
        )
        return Tensor(shape=list(tensor.shape), data=data)

    def _ensure_tensor_ints(self, tensor: Tensor, rule: str, location: SourceLocation) -> None:
        for entry in tensor.data.flat:
            self._expect_int(entry, rule, location)

    # Tensor built-ins
    def _shape(
        self,
        interpreter: "Interpreter",
        args: List[Value],
        __: List[Expression],
        ___: Environment,
        location: SourceLocation,
    ) -> Value:
        tensor = self._expect_tns(args[0], "SHAPE", location)
        shape_data = np.array([Value(TYPE_INT, dim) for dim in tensor.shape], dtype=object)
        return Value(TYPE_TNS, Tensor(shape=[len(shape_data)], data=shape_data))

    def _tlen(
        self,
        interpreter: "Interpreter",
        args: List[Value],
        __: List[Expression],
        ___: Environment,
        location: SourceLocation,
    ) -> Value:
        tensor = self._expect_tns(args[0], "TLEN", location)
        dim_index = self._expect_int(args[1], "TLEN", location)
        if dim_index <= 0 or dim_index > len(tensor.shape):
            raise ASMRuntimeError("TLEN dimension out of range", location=location, rewrite_rule="TLEN")
        return Value(TYPE_INT, tensor.shape[dim_index - 1])

    def _fill(
        self,
        interpreter: "Interpreter",
        args: List[Value],
        __: List[Expression],
        ___: Environment,
        location: SourceLocation,
    ) -> Value:
        tensor = self._expect_tns(args[0], "FILL", location)
        fill_value = args[1]
        if any(entry.type != fill_value.type for entry in tensor.data.flat):
            raise ASMRuntimeError("FILL value type must match existing tensor element types", location=location, rewrite_rule="FILL")
        new_data = np.array(
            [Value(fill_value.type, fill_value.value) for _ in range(tensor.data.size)],
            dtype=object,
        )
        return Value(TYPE_TNS, Tensor(shape=list(tensor.shape), data=new_data))

    def _tns(
        self,
        interpreter: "Interpreter",
        args: List[Value],
        __: List[Expression],
        ___: Environment,
        location: SourceLocation,
    ) -> Value:
        shape_val = self._expect_tns(args[0], "TNS", location)
        shape = self._shape_from_tensor(shape_val, "TNS", location)
        fill_value = args[1]
        tensor = interpreter._make_tensor_from_shape(shape, fill_value, "TNS", location)
        return Value(TYPE_TNS, tensor)

    def _madd(
        self,
        interpreter: "Interpreter",
        args: List[Value],
        __: List[Expression],
        ___: Environment,
        location: SourceLocation,
    ) -> Value:
        x = self._expect_tns(args[0], "MADD", location)
        y = self._expect_tns(args[1], "MADD", location)
        tensor = self._map_tensor_int_binary(x, y, "MADD", location, lambda a, b: a + b)
        return Value(TYPE_TNS, tensor)

    def _msub(
        self,
        interpreter: "Interpreter",
        args: List[Value],
        __: List[Expression],
        ___: Environment,
        location: SourceLocation,
    ) -> Value:
        x = self._expect_tns(args[0], "MSUB", location)
        y = self._expect_tns(args[1], "MSUB", location)
        tensor = self._map_tensor_int_binary(x, y, "MSUB", location, lambda a, b: a - b)
        return Value(TYPE_TNS, tensor)

    def _mmul(
        self,
        interpreter: "Interpreter",
        args: List[Value],
        __: List[Expression],
        ___: Environment,
        location: SourceLocation,
    ) -> Value:
        x = self._expect_tns(args[0], "MMUL", location)
        y = self._expect_tns(args[1], "MMUL", location)
        tensor = self._map_tensor_int_binary(x, y, "MMUL", location, lambda a, b: a * b)
        return Value(TYPE_TNS, tensor)

    def _mdiv(
        self,
        interpreter: "Interpreter",
        args: List[Value],
        __: List[Expression],
        ___: Environment,
        location: SourceLocation,
    ) -> Value:
        x = self._expect_tns(args[0], "MDIV", location)
        y = self._expect_tns(args[1], "MDIV", location)
        def _div(a: int, b: int) -> int:
            return self._safe_div(a, b)

        tensor = self._map_tensor_int_binary(x, y, "MDIV", location, _div)
        return Value(TYPE_TNS, tensor)

    def _msum(self, values: List[Value], location: SourceLocation) -> Value:
        tensors = [self._expect_tns(v, "MSUM", location) for v in values]
        acc = tensors[0]
        for tensor in tensors[1:]:
            acc = self._map_tensor_int_binary(acc, tensor, "MSUM", location, lambda a, b: a + b)
        return Value(TYPE_TNS, acc)

    def _mprod(self, values: List[Value], location: SourceLocation) -> Value:
        tensors = [self._expect_tns(v, "MPROD", location) for v in values]
        acc = tensors[0]
        for tensor in tensors[1:]:
            acc = self._map_tensor_int_binary(acc, tensor, "MPROD", location, lambda a, b: a * b)
        return Value(TYPE_TNS, acc)

    def _tadd(
        self,
        interpreter: "Interpreter",
        args: List[Value],
        __: List[Expression],
        ___: Environment,
        location: SourceLocation,
    ) -> Value:
        tensor = self._expect_tns(args[0], "TADD", location)
        scalar = self._expect_int(args[1], "TADD", location)
        result = self._map_tensor_int_scalar(tensor, scalar, "TADD", location, lambda a, b: a + b)
        return Value(TYPE_TNS, result)

    def _tsub(
        self,
        interpreter: "Interpreter",
        args: List[Value],
        __: List[Expression],
        ___: Environment,
        location: SourceLocation,
    ) -> Value:
        tensor = self._expect_tns(args[0], "TSUB", location)
        scalar = self._expect_int(args[1], "TSUB", location)
        result = self._map_tensor_int_scalar(tensor, scalar, "TSUB", location, lambda a, b: a - b)
        return Value(TYPE_TNS, result)

    def _tmul(
        self,
        interpreter: "Interpreter",
        args: List[Value],
        __: List[Expression],
        ___: Environment,
        location: SourceLocation,
    ) -> Value:
        tensor = self._expect_tns(args[0], "TMUL", location)
        scalar = self._expect_int(args[1], "TMUL", location)
        result = self._map_tensor_int_scalar(tensor, scalar, "TMUL", location, lambda a, b: a * b)
        return Value(TYPE_TNS, result)

    def _tdiv(
        self,
        interpreter: "Interpreter",
        args: List[Value],
        __: List[Expression],
        ___: Environment,
        location: SourceLocation,
    ) -> Value:
        tensor = self._expect_tns(args[0], "TDIV", location)
        scalar = self._expect_int(args[1], "TDIV", location)

        def _div(val: int, sc: int) -> int:
            return self._safe_div(val, sc)

        result = self._map_tensor_int_scalar(tensor, scalar, "TDIV", location, _div)
        return Value(TYPE_TNS, result)

    def _tpow(
        self,
        interpreter: "Interpreter",
        args: List[Value],
        __: List[Expression],
        ___: Environment,
        location: SourceLocation,
    ) -> Value:
        tensor = self._expect_tns(args[0], "TPOW", location)
        scalar = self._expect_int(args[1], "TPOW", location)
        result = self._map_tensor_int_scalar(tensor, scalar, "TPOW", location, lambda a, b: self._safe_pow(a, b))
        return Value(TYPE_TNS, result)

    def _cl(
        self,
        interpreter: "Interpreter",
        args: List[Value],
        __: List[Expression],
        ___: Environment,
        location: SourceLocation,
    ) -> Value:
        # Execute a shell command and return its exit code as INT.
        # Capture stdout/stderr so the REPL can display results without
        # spawning a visible console window on Windows.
        cmd = self._expect_str(args[0], "CL", location)
        cleanup_script: Optional[str] = None
        try:
            # Use text mode for captured output.
            run_kwargs: Dict[str, object] = {"stdout": subprocess.PIPE, "stderr": subprocess.PIPE, "text": True}
            # Default to shell=True to preserve existing behaviour for string commands.
            run_kwargs["shell"] = True

            # On Windows, avoid creating a visible console window for subprocesses.
            if platform.system().lower().startswith("win"):
                run_kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW

            command_to_run: Union[str, List[str]] = cmd
            if platform.system().lower().startswith("win") and len(cmd) > WINDOWS_COMMAND_LENGTH_LIMIT:
                stripped = cmd.lstrip()
                prefix = stripped.split(None, 1)[0].lower() if stripped else ""

                # If this is a PowerShell command, lift the -Command payload into
                # a temporary .ps1 to avoid CreateProcess length limits.
                if prefix in {"powershell", "pwsh"}:
                    ps_body = self._extract_powershell_body(cmd)
                    if ps_body is not None:
                        fd, script_path = tempfile.mkstemp(suffix=".ps1", text=True)
                        with os.fdopen(fd, "w", encoding="utf-8") as f:
                            f.write(ps_body)
                        cleanup_script = script_path
                        command_to_run = [prefix, "-NoProfile", "-File", script_path]
                        run_kwargs["shell"] = False
                    else:
                        # Fall back to the generic .cmd approach if parsing failed.
                        fd, script_path = tempfile.mkstemp(suffix=".cmd", text=True)
                        with os.fdopen(fd, "w", encoding="utf-8") as f:
                            f.write(cmd)
                        cleanup_script = script_path
                        command_to_run = ["cmd", "/c", script_path]
                        run_kwargs["shell"] = False
                else:
                    # Generic long-command fallback: write to .cmd and execute it.
                    fd, script_path = tempfile.mkstemp(suffix=".cmd", text=True)
                    with os.fdopen(fd, "w", encoding="utf-8") as f:
                        f.write(cmd)
                    cleanup_script = script_path
                    command_to_run = ["cmd", "/c", script_path]
                    run_kwargs["shell"] = False

            completed = subprocess.run(command_to_run, **run_kwargs)
            code = completed.returncode
            out = completed.stdout or ""
            err = completed.stderr or ""
        except OSError as exc:
            raise ASMRuntimeError(f"CL failed: {exc}", location=location, rewrite_rule="CL")
        finally:
            if cleanup_script:
                try:
                    os.unlink(cleanup_script)
                except OSError:
                    pass

        # Record the CL event for deterministic logging/replay, including captured output.
        interpreter.io_log.append({"event": "CL", "cmd": cmd, "code": code, "stdout": out, "stderr": err})

        # Forward captured output to the interpreter's output sink so REPL users see it.
        if out:
            if not interpreter.shushed:
                interpreter.output_sink(out)
        if err:
            # Send stderr to the same sink; callers can choose how to display it.
            if not interpreter.shushed:
                interpreter.output_sink(err)

        # Normalize return to INT
        return Value(TYPE_INT, int(code))

    def _shush(
        self,
        interpreter: "Interpreter",
        args: List[Value],
        __: List[Expression],
        ___: Environment,
        ____: SourceLocation,
    ) -> Value:
        """SHUSH(): suppress forwarding console output (PRINT, CL, etc.).
        INPUT prompts and RUN-invoked output remain visible.
        """
        interpreter.shushed = True
        return Value(TYPE_INT, 0)

    def _unshush(
        self,
        interpreter: "Interpreter",
        args: List[Value],
        __: List[Expression],
        ___: Environment,
        ____: SourceLocation,
    ) -> Value:
        """UNSHUSH(): re-enable forwarding of console output.
        Clears the shush flag so subsequent PRINT/CL outputs are forwarded
        to the configured output sink. Returns INT 0.
        """
        interpreter.shushed = False
        return Value(TYPE_INT, 0)

    def _exit(
        self,
        interpreter: "Interpreter",
        args: List[Value],
        __: List[Expression],
        ___: Environment,
        ____: SourceLocation,
    ) -> Value:
        code_val = args[0] if args else Value(TYPE_INT, 0)
        code = self._expect_int(code_val, "EXIT", ____)
        interpreter.io_log.append({"event": "EXIT", "code": code})
        raise ExitSignal(code)


class Interpreter:
    def __init__(
        self,
        *,
        source: str,
        filename: str,
        verbose: bool,
        input_provider: Optional[Callable[[], str]] = None,
        output_sink: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.source = source
        normalized_filename = filename if filename == "<string>" else os.path.abspath(filename)
        self.filename = normalized_filename
        self.entry_filename = normalized_filename
        self.verbose = verbose
        self.input_provider = input_provider or (lambda: input(">>> "))
        self.output_sink = output_sink or (lambda text: print(text))
        self.builtins = Builtins()
        self.functions: Dict[str, Function] = {}
        self.logger = StateLogger(verbose=verbose)
        self.logger.record(frame=None, location=None, statement="<seed>", rewrite_record={"rule": "SEED"})
        self.io_log: List[Dict[str, Any]] = []
        self.call_stack: List[Frame] = []
        self.frame_counter = 0
        # When true, suppress forwarding of console output (PRINT, CL, etc.).
        # INPUT prompts and output produced via RUN are still forwarded.
        self.shushed: bool = False
        # Cache imported modules so repeated IMPORTs return the same
        # namespace/instance rather than re-executing the module.
        # Keys are module identifiers (the name passed to IMPORT).
        self.module_cache: Dict[str, Environment] = {}
        # Keep any function objects created for modules so they can be
        # re-registered if necessary without re-running module code.
        self.module_functions: Dict[str, List[Function]] = {}

    def parse(self) -> Program:
        lexer = Lexer(self.source, self.filename)
        tokens = lexer.tokenize()
        parser = Parser(tokens, self.filename, self.source.splitlines())
        return parser.parse()

    def run(self) -> None:
        program = self.parse()
        global_env = Environment()
        global_frame = self._new_frame("<top-level>", global_env, None)
        self.call_stack.append(global_frame)
        try:
            self._execute_block(program.statements, global_env)
        except BreakSignal as bs:
            raise ASMRuntimeError(f"BREAK({bs.count}) escaped enclosing loops", rewrite_rule="BREAK")
        except ContinueSignal:
            raise ASMRuntimeError("CONTINUE used outside loop", rewrite_rule="CONTINUE")
        except ASMRuntimeError as error:
            if self.logger.entries:
                error.step_index = self.logger.entries[-1].step_index
            raise
        else:
            self.call_stack.pop()

    def _execute_block(self, statements: List[Statement], env: Environment) -> None:
        i = 0
        frame: Frame = self.call_stack[-1]
        gotopoints: Dict[int, int] = frame.gotopoints
        while i < len(statements):
            statement = statements[i]
            if isinstance(statement, GotopointStatement):
                self._log_step(rule=statement.__class__.__name__, location=statement.location)
                gid = self._evaluate_expression(statement.expression, env)
                if gid.type == TYPE_INT:
                    if gid.value < 0:
                        raise ASMRuntimeError(
                            "GOTOPOINT id must be non-negative",
                            location=statement.location,
                            rewrite_rule="GOTOPOINT",
                        )
                    key = (TYPE_INT, gid.value)
                elif gid.type == TYPE_STR:
                    key = (TYPE_STR, gid.value)
                else:
                    raise ASMRuntimeError(
                        "GOTOPOINT expects int or string identifier",
                        location=statement.location,
                        rewrite_rule="GOTOPOINT",
                    )
                gotopoints[key] = i
                i += 1
                continue
            try:
                self._execute_statement(statement, env)
            except JumpSignal as js:
                target = js.target
                key = (target.type, target.value)
                if key not in gotopoints:
                    raise ASMRuntimeError(
                        f"GOTO to undefined gotopoint '{target.value}'", location=statement.location, rewrite_rule="GOTO"
                    )
                i = gotopoints[key]
                continue
            i += 1

    def _execute_statement(self, statement: Statement, env: Environment) -> None:
        self._log_step(rule=statement.__class__.__name__, location=statement.location)
        if isinstance(statement, Assignment):
            if statement.target in self.functions:
                raise ASMRuntimeError(
                    f"Identifier '{statement.target}' already bound as function", location=statement.location, rewrite_rule="ASSIGN"
                )
            value = self._evaluate_expression(statement.expression, env)
            env.set(statement.target, value, declared_type=statement.declared_type)
            return
        if isinstance(statement, TensorSetStatement):
            base_expr, index_nodes = self._gather_index_chain(statement.target)
            if not isinstance(base_expr, Identifier):
                raise ASMRuntimeError(
                    "Indexed assignment requires identifier base",
                    location=statement.location,
                    rewrite_rule="ASSIGN",
                )

            # Respect identifier freezing: element assignment is still a form of reassignment.
            env_found = env._find_env(base_expr.name)
            if env_found is not None and (base_expr.name in env_found.frozen or base_expr.name in env_found.permafrozen):
                raise ASMRuntimeError(
                    f"Identifier '{base_expr.name}' is frozen and cannot be reassigned",
                    location=statement.location,
                    rewrite_rule="ASSIGN",
                )

            base_val = self._evaluate_expression(base_expr, env)
            if base_val.type != TYPE_TNS:
                raise ASMRuntimeError(
                    "Indexed assignment requires a tensor base",
                    location=statement.location,
                    rewrite_rule="ASSIGN",
                )
            assert isinstance(base_val.value, Tensor)
            indices = [self._expect_int(self._evaluate_expression(node, env), "ASSIGN", statement.location) for node in index_nodes]
            new_value = self._evaluate_expression(statement.value, env)
            self._mutate_tensor_value(base_val.value, indices, new_value, statement.location)
            return
        if isinstance(statement, ExpressionStatement):
            self._evaluate_expression(statement.expression, env)
            return
        if isinstance(statement, IfStatement):
            self._execute_if(statement, env)
            return
        if isinstance(statement, WhileStatement):
            self._execute_while(statement, env)
            return
        if isinstance(statement, ForStatement):
            self._execute_for(statement, env)
            return
        if isinstance(statement, FuncDef):
            if statement.name in self.builtins.table:
                raise ASMRuntimeError(
                    f"Function name '{statement.name}' conflicts with built-in", location=statement.location
                )
            self.functions[statement.name] = Function(
                name=statement.name,
                params=statement.params,
                return_type=statement.return_type,
                body=statement.body,
                closure=env,
            )
            return
        if isinstance(statement, PopStatement):
            frame: Frame = self.call_stack[-1]
            if frame.name == "<top-level>":
                raise ASMRuntimeError("POP outside of function", location=statement.location, rewrite_rule="POP")
            # Expect identifier expression to delete a symbol
            expr = statement.expression
            if not isinstance(expr, Identifier):
                raise ASMRuntimeError("POP expects identifier", location=statement.location, rewrite_rule="POP")
            name = expr.name
            try:
                value = env.get(name)
            except ASMRuntimeError as err:
                err.location = statement.location
                raise
            try:
                env.delete(name)
            except ASMRuntimeError as err:
                err.location = statement.location
                raise
            raise ReturnSignal(value)
        if isinstance(statement, ReturnStatement):
            frame: Frame = self.call_stack[-1]
            if frame.name == "<top-level>":
                raise ASMRuntimeError("RETURN outside of function", location=statement.location, rewrite_rule="RETURN")
            if statement.expression:
                value = self._evaluate_expression(statement.expression, env)
            else:
                fn = self.functions.get(frame.name)
                if fn is None:
                    value = Value(TYPE_INT, 0)
                elif fn.return_type == TYPE_INT:
                    value = Value(TYPE_INT, 0)
                elif fn.return_type == TYPE_STR:
                    value = Value(TYPE_STR, "")
                elif fn.return_type == TYPE_TNS:
                    raise ASMRuntimeError(
                        "TNS functions must RETURN a tensor value",
                        location=statement.location,
                        rewrite_rule="RETURN",
                    )
                else:
                    value = Value(TYPE_INT, 0)
            raise ReturnSignal(value)
        if isinstance(statement, BreakStatement):
            count_val = self._evaluate_expression(statement.expression, env)
            count = self._expect_int(count_val, "BREAK", statement.location)
            if count <= 0:
                raise ASMRuntimeError("BREAK count must be > 0", location=statement.location, rewrite_rule="BREAK")
            raise BreakSignal(count)
        if isinstance(statement, ContinueStatement):
            # Signal to the innermost loop to skip to next iteration.
            raise ContinueSignal()
        if isinstance(statement, GotoStatement):
            target = self._evaluate_expression(statement.expression, env)
            raise JumpSignal(target)
        raise ASMRuntimeError("Unsupported statement", location=statement.location)

    def _execute_if(self, statement: IfStatement, env: Environment) -> None:
        if self._condition_int(self._evaluate_expression(statement.condition, env), statement.location) != 0:
            self._execute_block(statement.then_block.statements, env)
            return
        for branch in statement.elifs:
            if self._condition_int(self._evaluate_expression(branch.condition, env), branch.condition.location) != 0:
                self._execute_block(branch.block.statements, env)
                return
        if statement.else_block:
            self._execute_block(statement.else_block.statements, env)

    def _execute_while(self, statement: WhileStatement, env: Environment) -> None:
        while self._condition_int(self._evaluate_expression(statement.condition, env), statement.condition.location) != 0:
            try:
                self._execute_block(statement.block.statements, env)
            except BreakSignal as bs:
                if bs.count > 1:
                    bs.count -= 1
                    raise
                return
            except ContinueSignal:
                # Evaluate condition now to determine if there will be another iteration.
                try:
                    next_cond_val = self._evaluate_expression(statement.condition, env)
                    next_cond = self._condition_int(next_cond_val, statement.condition.location)
                except ASMRuntimeError:
                    # Propagate evaluation errors
                    raise
                if next_cond != 0:
                    # proceed to next iteration
                    continue
                # no next iteration -> behave like BREAK(1)
                return

    def _execute_for(self, statement: ForStatement, env: Environment) -> None:
        target_val = self._evaluate_expression(statement.target_expr, env)
        target = self._expect_int(target_val, "FOR", statement.location)
        env.set(statement.counter, Value(TYPE_INT, 0), declared_type=TYPE_INT)
        while self._expect_int(env.get(statement.counter), "FOR", statement.location) < target:
            try:
                self._execute_block(statement.block.statements, env)
            except BreakSignal as bs:
                if bs.count > 1:
                    bs.count -= 1
                    raise
                return
            except ContinueSignal:
                # For FOR loops, increment the counter and decide whether to continue
                current = self._expect_int(env.get(statement.counter), "FOR", statement.location)
                if current + 1 < target:
                    env.set(statement.counter, Value(TYPE_INT, current + 1))
                    continue
                # no further iterations -> behave like BREAK(1)
                return
            env.set(statement.counter, Value(TYPE_INT, self._expect_int(env.get(statement.counter), "FOR", statement.location) + 1))

    def _condition_int(self, value: Value, location: Optional[SourceLocation]) -> int:
        if value.type == TYPE_INT:
            return value.value  # type: ignore[return-value]
        if value.type == TYPE_STR:
            text = value.value  # type: ignore[assignment]
            if text == "":
                return 0
            if set(text).issubset({"0", "1"}):
                return int(text, 2)
            return 1
        if value.type == TYPE_TNS:
            assert isinstance(value.value, Tensor)
            return 1 if self._tensor_truthy(value.value) else 0
        raise ASMRuntimeError("Unsupported type in condition", location=location)

    def _expect_int(self, value: Value, rule: str, location: Optional[SourceLocation]) -> int:
        if value.type != TYPE_INT:
            raise ASMRuntimeError(f"{rule} expects integer value", location=location, rewrite_rule=rule)
        return value.value  # type: ignore[return-value]

    def _tensor_total_size(self, shape: List[int]) -> int:
        size = 1
        for dim in shape:
            size *= dim
        return size

    def _tensor_truthy(self, tensor: Tensor) -> bool:
        for item in tensor.data.flat:
            if item.type == TYPE_INT and item.value != 0:
                return True
            if item.type == TYPE_STR and item.value != "":
                return True
            if item.type == TYPE_TNS and self._tensor_truthy(item.value):
                return True
        return False

    def _values_equal(self, left: Value, right: Value) -> bool:
        if left.type != right.type:
            return False
        if left.type == TYPE_TNS:
            assert isinstance(left.value, Tensor)
            assert isinstance(right.value, Tensor)
            return self._tensor_equal(left.value, right.value)
        return left.value == right.value

    def _tensor_equal(self, left: Tensor, right: Tensor) -> bool:
        if left.shape != right.shape:
            return False
        return all(self._values_equal(a, b) for a, b in zip(left.data.flat, right.data.flat))

    def _validate_tensor_shape(self, shape: List[int], rule: str, location: SourceLocation) -> None:
        if not shape:
            raise ASMRuntimeError("Tensor shape must have at least one dimension", location=location, rewrite_rule=rule)
        for dim in shape:
            if dim <= 0:
                raise ASMRuntimeError("Tensor dimensions must be positive", location=location, rewrite_rule=rule)

    def _tensor_flat_index(self, tensor: Tensor, indices: List[int], rule: str, location: SourceLocation) -> int:
        if len(indices) != len(tensor.shape):
            raise ASMRuntimeError("Incorrect number of tensor indices", location=location, rewrite_rule=rule)
        offset = 0
        stride = 1
        for dim_len, raw in zip(reversed(tensor.shape), reversed(indices)):
            idx = raw
            if idx == 0:
                raise ASMRuntimeError("Tensor indices are 1-indexed", location=location, rewrite_rule=rule)
            if idx < 0:
                idx = dim_len + idx + 1
            if idx <= 0 or idx > dim_len:
                raise ASMRuntimeError("Tensor index out of range", location=location, rewrite_rule=rule)
            offset += (idx - 1) * stride
            stride *= dim_len
        return offset

    def _make_tensor_from_shape(self, shape: List[int], fill_value: Value, rule: str, location: SourceLocation) -> Tensor:
        self._validate_tensor_shape(shape, rule, location)
        total = self._tensor_total_size(shape)
        # Duplicate scalar values to avoid accidental aliasing when tensors are nested.
        def _clone(val: Value) -> Value:
            if val.type == TYPE_TNS:
                assert isinstance(val.value, Tensor)
                return Value(TYPE_TNS, val.value)
            return Value(val.type, val.value)

        data = np.array([_clone(fill_value) for _ in range(total)], dtype=object)
        return Tensor(shape=list(shape), data=data)

    def _set_tensor_value(self, tensor: Tensor, indices: List[int], value: Value, location: SourceLocation) -> Tensor:
        offset = self._tensor_flat_index(tensor, indices, "ASSIGN", location)
        current = tensor.data[offset]
        if current.type != value.type:
            raise ASMRuntimeError("Tensor element type mismatch", location=location, rewrite_rule="ASSIGN")
        new_data = tensor.data.copy()
        new_data[offset] = value
        return Tensor(shape=list(tensor.shape), data=new_data)

    def _mutate_tensor_value(self, tensor: Tensor, indices: List[int], value: Value, location: SourceLocation) -> None:
        offset = self._tensor_flat_index(tensor, indices, "ASSIGN", location)
        current = tensor.data[offset]
        if current.type != value.type:
            raise ASMRuntimeError("Tensor element type mismatch", location=location, rewrite_rule="ASSIGN")
        tensor.data[offset] = value

    def _build_tensor_from_literal(self, literal: TensorLiteral, env: Environment) -> Tensor:
        items = literal.items
        if not items:
            raise ASMRuntimeError("Tensor literal cannot be empty", location=literal.location, rewrite_rule="TNS")
        flat: List[Value] = []
        subshape: Optional[List[int]] = None
        for item in items:
            if isinstance(item, TensorLiteral):
                nested = self._build_tensor_from_literal(item, env)
                if subshape is None:
                    subshape = nested.shape
                elif subshape != nested.shape:
                    raise ASMRuntimeError("Inconsistent tensor shape", location=item.location, rewrite_rule="TNS")
                flat.extend(list(nested.data.flat))
            else:
                val = self._evaluate_expression(item, env)
                if subshape is None:
                    subshape = []
                elif subshape != []:
                    raise ASMRuntimeError("Inconsistent tensor shape", location=item.location, rewrite_rule="TNS")
                flat.append(val)
        shape = [len(items)] + (subshape or [])
        self._validate_tensor_shape(shape, "TNS", literal.location)
        expected = self._tensor_total_size(shape)
        if len(flat) != expected:
            raise ASMRuntimeError("Tensor literal size mismatch", location=literal.location, rewrite_rule="TNS")
        return Tensor(shape=shape, data=np.array(flat, dtype=object))

    def _gather_index_chain(self, expr: Expression) -> Tuple[Expression, List[Expression]]:
        indices: List[Expression] = []
        current = expr
        while isinstance(current, IndexExpression):
            indices = current.indices + indices
            current = current.base
        return current, indices

    def _evaluate_expression(self, expression: Expression, env: Environment) -> Value:
        if isinstance(expression, Literal):
            return Value(expression.literal_type, expression.value)
        if isinstance(expression, TensorLiteral):
            tensor = self._build_tensor_from_literal(expression, env)
            return Value(TYPE_TNS, tensor)
        if isinstance(expression, IndexExpression):
            base_expr, index_nodes = self._gather_index_chain(expression)
            base_val = self._evaluate_expression(base_expr, env)
            if base_val.type != TYPE_TNS:
                raise ASMRuntimeError("Indexed access requires a tensor", location=expression.location, rewrite_rule="INDEX")
            assert isinstance(base_val.value, Tensor)
            indices = [self._expect_int(self._evaluate_expression(idx, env), "INDEX", expression.location) for idx in index_nodes]
            offset = self._tensor_flat_index(base_val.value, indices, "INDEX", expression.location)
            return base_val.value.data[offset]
        if isinstance(expression, Identifier):
            try:
                return env.get(expression.name)
            except ASMRuntimeError as err:
                err.location = expression.location
                if expression.name == "INPUT":
                    result = self.builtins.invoke(self, "INPUT", [], [], env, expression.location)
                    self._log_step(rule="INPUT", location=expression.location, extra={"args": [], "result": result.value})
                    return result
                raise
        if isinstance(expression, CallExpression):
            if expression.name == "IMPORT":
                if any(arg.name for arg in expression.args):
                    raise ASMRuntimeError("IMPORT does not accept keyword arguments", location=expression.location, rewrite_rule="IMPORT")
                first_expr = expression.args[0].expression if expression.args else None
                module_label = first_expr.name if isinstance(first_expr, Identifier) else None
                dummy_args: List[Value] = [Value(TYPE_INT, 0)] * len(expression.args)
                arg_nodes = [arg.expression for arg in expression.args]
                try:
                    result = self.builtins.invoke(self, expression.name, dummy_args, arg_nodes, env, expression.location)
                except ASMRuntimeError:
                    self._log_step(rule="IMPORT", location=expression.location, extra={"module": module_label, "status": "error"})
                    raise
                self._log_step(rule="IMPORT", location=expression.location, extra={"module": module_label, "result": result.value})
                return result
            if expression.name in ("DEL", "EXIST"):
                if any(arg.name for arg in expression.args):
                    raise ASMRuntimeError(
                        f"{expression.name} does not accept keyword arguments",
                        location=expression.location,
                        rewrite_rule=expression.name,
                    )
                dummy_args: List[Value] = [Value(TYPE_INT, 0)] * len(expression.args)
                arg_nodes = [arg.expression for arg in expression.args]
                try:
                    result = self.builtins.invoke(self, expression.name, dummy_args, arg_nodes, env, expression.location)
                except ASMRuntimeError:
                    self._log_step(rule=expression.name, location=expression.location, extra={"args": None, "status": "error"})
                    raise
                self._log_step(rule=expression.name, location=expression.location, extra={"args": None, "result": result.value})
                return result
            positional_args: List[Value] = []
            keyword_args: Dict[str, Value] = {}
            for arg in expression.args:
                value = self._evaluate_expression(arg.expression, env)
                if arg.name is None:
                    positional_args.append(value)
                else:
                    if arg.name in keyword_args:
                        raise ASMRuntimeError(
                            f"Duplicate keyword argument '{arg.name}'",
                            location=expression.location,
                            rewrite_rule=expression.name,
                        )
                    keyword_args[arg.name] = value
            func_name: Optional[str] = None
            if expression.name in self.functions:
                func_name = expression.name
            else:
                if self.call_stack:
                    current = self.call_stack[-1]
                    if "." in current.name:
                        module_prefix = current.name.split(".", 1)[0]
                        candidate = f"{module_prefix}.{expression.name}"
                        if candidate in self.functions:
                            func_name = candidate
            if func_name is not None:
                self._log_step(
                    rule="CALL",
                    location=expression.location,
                    extra={
                        "function": func_name,
                        "positional": [a.value for a in positional_args],
                        "keyword": {k: v.value for k, v in keyword_args.items()},
                    },
                )
                return self._call_user_function(
                    self.functions[func_name],
                    positional_args,
                    keyword_args,
                    expression.location,
                )
            try:
                if keyword_args:
                    if expression.name in {"READFILE", "WRITEFILE"}:
                        allowed = {"coding"}
                        unexpected = [k for k in keyword_args if k not in allowed]
                        if unexpected:
                            raise ASMRuntimeError(
                                f"Unexpected keyword arguments: {', '.join(sorted(unexpected))}",
                                location=expression.location,
                                rewrite_rule=expression.name,
                            )
                        if "coding" in keyword_args:
                            positional_args.append(keyword_args.pop("coding"))
                    if keyword_args:
                        raise ASMRuntimeError(
                            f"{expression.name} does not accept keyword arguments",
                            location=expression.location,
                            rewrite_rule=expression.name,
                        )
                arg_nodes = [a.expression for a in expression.args]
                result = self.builtins.invoke(self, expression.name, positional_args, arg_nodes, env, expression.location)
            except ASMRuntimeError:
                self._log_step(
                    rule=expression.name,
                    location=expression.location,
                    extra={
                        "args": [a.value for a in positional_args],
                        "keyword": {k: v.value for k, v in keyword_args.items()},
                        "status": "error",
                    },
                )
                raise
            self._log_step(
                rule=expression.name,
                location=expression.location,
                extra={
                    "args": [a.value for a in positional_args],
                    "keyword": {k: v.value for k, v in keyword_args.items()},
                    "result": result.value,
                },
            )
            return result
        raise ASMRuntimeError("Unsupported expression", location=expression.location)

    def _call_user_function(
        self,
        function: Function,
        positional_args: List[Value],
        keyword_args: Dict[str, Value],
        call_location: SourceLocation,
    ) -> Value:
        if len(positional_args) > len(function.params):
            raise ASMRuntimeError(
                f"Function {function.name} expects at most {len(function.params)} positional arguments but received {len(positional_args)}",
                location=call_location,
                rewrite_rule=function.name,
            )

        env = Environment(parent=function.closure)

        kwds = dict(keyword_args)

        for param, arg in zip(function.params, positional_args):
            if arg.type != param.type:
                raise ASMRuntimeError(
                    f"Argument for '{param.name}' expected {param.type} but got {arg.type}",
                    location=call_location,
                    rewrite_rule=function.name,
                )
            env.set(param.name, arg, declared_type=param.type)

        remaining_params = function.params[len(positional_args) :]
        for param in remaining_params:
            if param.name in kwds:
                if param.default is None:
                    raise ASMRuntimeError(
                        f"Parameter '{param.name}' does not accept keyword arguments",
                        location=call_location,
                        rewrite_rule=function.name,
                    )
                value = kwds.pop(param.name)
                if value.type != param.type:
                    raise ASMRuntimeError(
                        f"Argument for '{param.name}' expected {param.type} but got {value.type}",
                        location=call_location,
                        rewrite_rule=function.name,
                    )
                env.set(param.name, value, declared_type=param.type)
                continue
            if param.default is None:
                raise ASMRuntimeError(
                    f"Missing required argument '{param.name}' for function {function.name}",
                    location=call_location,
                    rewrite_rule=function.name,
                )
            default_value = self._evaluate_expression(param.default, env)
            if default_value.type != param.type:
                raise ASMRuntimeError(
                    f"Default for '{param.name}' expected {param.type} but got {default_value.type}",
                    location=call_location,
                    rewrite_rule=function.name,
                )
            env.set(param.name, default_value, declared_type=param.type)

        if kwds:
            unexpected = ", ".join(sorted(kwds.keys()))
            raise ASMRuntimeError(
                f"Unexpected keyword arguments: {unexpected}",
                location=call_location,
                rewrite_rule=function.name,
            )
        frame = self._new_frame(function.name, env, call_location)
        self.call_stack.append(frame)
        try:
            self._execute_block(function.body.statements, env)
        except ReturnSignal as signal:
            self.call_stack.pop()
            if signal.value.type != function.return_type:
                raise ASMRuntimeError(
                    f"Function {function.name} must return {function.return_type} but got {signal.value.type}",
                    location=call_location,
                    rewrite_rule=function.name,
                )
            return signal.value
        except ASMRuntimeError:
            raise
        else:
            self.call_stack.pop()
            if function.return_type == TYPE_INT:
                return Value(TYPE_INT, 0)
            if function.return_type == TYPE_STR:
                return Value(TYPE_STR, "")
            if function.return_type == TYPE_TNS:
                raise ASMRuntimeError(
                    f"Function {function.name} must return a tensor value",
                    location=call_location,
                    rewrite_rule=function.name,
                )
            return Value(TYPE_INT, 0)

    def _new_frame(self, name: str, env: Environment, call_location: Optional[SourceLocation]) -> Frame:
        frame_id = f"f_{self.frame_counter:04d}"
        self.frame_counter += 1
        return Frame(name=name, env=env, frame_id=frame_id, call_location=call_location)

    def _log_step(
        self,
        *,
        rule: str,
        location: Optional[SourceLocation],
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        frame = self.call_stack[-1] if self.call_stack else None
        env_snapshot = frame.env.snapshot() if (self.verbose and frame) else None
        statement = location.statement if location else None
        rewrite = {"rule": rule}
        if extra:
            rewrite.update(extra)
        self.logger.record(
            frame=frame,
            location=location,
            statement=statement,
            env_snapshot=env_snapshot,
            rewrite_record=rewrite,
        )


@dataclass
class TracebackFrame:
    name: str
    location: Optional[SourceLocation]
    statement: Optional[str]
    state_entry: Optional[StateEntry]


class TracebackFormatter:
    def __init__(self, interpreter: Interpreter) -> None:
        self.interpreter = interpreter

    def build_frames(self) -> List[TracebackFrame]:
        frames: List[TracebackFrame] = []
        for frame in self.interpreter.call_stack:
            entry = self.interpreter.logger.last_entry_for_frame(frame.frame_id)
            location = entry.source_location if entry else frame.call_location
            frames.append(
                TracebackFrame(
                    name=frame.name,
                    location=location,
                    statement=entry.statement if entry else None,
                    state_entry=entry,
                )
            )
        return frames

    def format_text(self, error: ASMRuntimeError, verbose: bool) -> str:
        lines = ["Traceback (most recent call last):"]
        for frame in self.build_frames():
            if frame.location:
                lines.append(f"  File \"{frame.location.file}\", line {frame.location.line}, in {frame.name}")
                if frame.statement:
                    lines.append(f"    {frame.statement}")
            else:
                lines.append(f"  <unknown location> in {frame.name}")
            if frame.state_entry:
                lines.append(
                    f"    State log index: {frame.state_entry.step_index}  State id: {frame.state_entry.state_id}"
                )
                if verbose and frame.state_entry.env_snapshot is not None:
                    snapshot = ", ".join(f"{k}={v}" for k, v in frame.state_entry.env_snapshot.items())
                    lines.append(f"    Env snapshot: {snapshot}")
        rule = error.rewrite_rule or "runtime"
        lines.append(f"{error.__class__.__name__}: {error.message} (rewrite: {rule})")
        return "\n".join(lines)

    def to_json(self, error: ASMRuntimeError) -> str:
        frames_json: List[Dict[str, Any]] = []
        for index, frame in enumerate(self.build_frames()):
            entry: Dict[str, Any] = {"frame_index": index, "name": frame.name}
            if frame.location:
                entry["source_location"] = {
                    "file": frame.location.file,
                    "line": frame.location.line,
                    "statement": frame.location.statement,
                }
            if frame.state_entry:
                entry["state_id"] = frame.state_entry.state_id
                entry["step_index"] = frame.state_entry.step_index
                if frame.state_entry.env_snapshot is not None:
                    entry["env_snapshot"] = frame.state_entry.env_snapshot
                if frame.state_entry.rewrite_record is not None:
                    entry["rewrite_record"] = frame.state_entry.rewrite_record
            frames_json.append(entry)
        data = {
            "error": {
                "type": error.__class__.__name__,
                "message": error.message,
                "failing_step_index": error.step_index,
            },
            "traceback": frames_json,
        }
        return json.dumps(data, indent=2)
