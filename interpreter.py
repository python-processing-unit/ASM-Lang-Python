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
from extensions import ASMExtensionError, HookRegistry, RuntimeServices, StepContext, TypeContext, TypeRegistry, TypeSpec, build_default_services
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
TYPE_FLT = "FLT"
TYPE_STR = "STR"
TYPE_TNS = "TNS"

# On Windows, command lines over a certain length cause CreateProcess errors
WINDOWS_COMMAND_LENGTH_LIMIT = 8000

@dataclass(frozen=True)
class Tensor:
    shape: List[int]
    data: NDArray[Any]
    strides: Tuple[int, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        # Cache row-major strides for fast flat indexing.
        # For shape [d0, d1, ..., dn-1], strides are
        # [d1*d2*...*dn-1, d2*...*dn-1, ..., 1].
        stride = 1
        out: List[int] = [0] * len(self.shape)
        for i in range(len(self.shape) - 1, -1, -1):
            out[i] = stride
            stride *= int(self.shape[i])
        object.__setattr__(self, "strides", tuple(out))


@dataclass
class Value:
    type: str
    value: Any


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

    def get_optional(self, name: str) -> Optional[Value]:
        env = self._find_env(name)
        if env is not None:
            return env.values[name]
        return None

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
            try:
                rendered = str(val.value)
            except Exception:
                rendered = repr(val.value)
            if len(rendered) > 80:
                rendered = rendered[:77] + "..."
            return f"{val.type}:{rendered}"

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
        # Hot-path: reuse the caller-provided dict rather than copying.
        # The interpreter constructs a fresh dict per step in _log_step, so
        # sharing is safe and avoids one allocation per step.
        rewrite = {} if rewrite_record is None else rewrite_record
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
        # Numeric operators that support INT and FLT (no mixing).
        self._register_custom("ADD", 2, 2, self._add)
        self._register_custom("SUB", 2, 2, self._sub)
        self._register_custom("MUL", 2, 2, self._mul)
        self._register_custom("DIV", 2, 2, self._div)
        self._register_int_only("CDIV", 2, self._safe_cdiv)
        self._register_custom("MOD", 2, 2, self._mod)
        self._register_custom("POW", 2, 2, self._pow)
        self._register_custom("ROOT", 2, 2, self._root)
        self._register_custom("NEG", 1, 1, self._neg)
        self._register_custom("ABS", 1, 1, self._abs)
        self._register_custom("GCD", 2, 2, self._gcd)
        self._register_custom("LCM", 2, 2, self._lcm_num)
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
        self._register_custom("GT", 2, 2, self._gt)
        self._register_custom("LT", 2, 2, self._lt)
        self._register_custom("GTE", 2, 2, self._gte)
        self._register_custom("LTE", 2, 2, self._lte)
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
        self._register_custom("LOG", 1, 1, self._log)
        self._register_int_only("CLOG", 1, self._safe_clog)
        self._register_custom("INT", 1, 1, self._int_op)
        self._register_custom("FLT", 1, 1, self._flt_op)
        self._register_custom("STR", 1, 1, self._str_op)
        self._register_custom("UPPER", 1, 1, self._upper)
        self._register_custom("LOWER", 1, 1, self._lower)
        self._register_custom("STRIP", 2, 2, self._strip)
        self._register_custom("REPLACE", 3, 3, self._replace)
        self._register_custom("MAIN", 0, 0, self._main)
        self._register_custom("OS", 0, 0, self._os)
        self._register_custom("IMPORT", 1, 2, self._import)
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
        self._register_custom("ISFLT", 1, 1, self._isflt)
        self._register_custom("ISSTR", 1, 1, self._isstr)
        self._register_custom("ISTNS", 1, 1, self._istns)
        self._register_custom("TYPE", 1, 1, self._type)
        self._register_custom("ROUND", 1, 3, self._round)
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
        self._register_custom("CONVOLVE", 2, 2, self._convolve)
        self._register_custom("FLIP", 1, 1, self._flip)
        self._register_custom("TFLIP", 2, 2, self._tflip)

    def _register_int_only(self, name: str, arity: int, func: Callable[..., int]) -> None:
        def impl(_: "Interpreter", args: List[Value], __: List[Expression], ___: Environment, location: SourceLocation) -> Value:
            ints = [self._expect_int(arg, name, location) for arg in args]
            return Value(TYPE_INT, func(*ints))

        self.table[name] = BuiltinFunction(name=name, min_args=arity, max_args=arity, impl=impl)

    def _register_variadic(
        self,
        name: str,
        min_args: int,
        func: Callable[["Interpreter", List[Value], SourceLocation], Value],
    ) -> None:
        def impl(interpreter: "Interpreter", args: List[Value], __: List[Expression], ___: Environment, location: SourceLocation) -> Value:
            return func(interpreter, args, location)

        self.table[name] = BuiltinFunction(name=name, min_args=min_args, max_args=None, impl=impl)

    def _register_custom(
        self,
        name: str,
        min_args: int,
        max_args: Optional[int],
        impl: BuiltinImpl,
    ) -> None:
        self.table[name] = BuiltinFunction(name=name, min_args=min_args, max_args=max_args, impl=impl)

    def register_extension_operator(
        self,
        *,
        name: str,
        min_args: int,
        max_args: Optional[int],
        impl: BuiltinImpl,
    ) -> None:
        if name in self.table:
            raise ASMExtensionError(f"Cannot override existing operator '{name}'")
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

    def _expect_flt(self, value: Value, rule: str, location: SourceLocation) -> float:
        if value.type != TYPE_FLT:
            raise ASMRuntimeError(f"{rule} expects float arguments", location=location, rewrite_rule=rule)
        assert isinstance(value.value, float)
        return value.value

    def _expect_num_pair(self, args: List[Value], rule: str, location: SourceLocation) -> Tuple[str, Any, Any]:
        if len(args) != 2:
            raise ASMRuntimeError(f"{rule} expects 2 arguments", location=location, rewrite_rule=rule)
        a, b = args[0], args[1]
        if a.type != b.type:
            raise ASMRuntimeError(f"{rule} cannot mix INT and FLT", location=location, rewrite_rule=rule)
        if a.type == TYPE_INT:
            return TYPE_INT, self._expect_int(a, rule, location), self._expect_int(b, rule, location)
        if a.type == TYPE_FLT:
            return TYPE_FLT, self._expect_flt(a, rule, location), self._expect_flt(b, rule, location)
        raise ASMRuntimeError(f"{rule} expects INT or FLT arguments", location=location, rewrite_rule=rule)

    def _expect_num_unary(self, args: List[Value], rule: str, location: SourceLocation) -> Tuple[str, Any]:
        if len(args) != 1:
            raise ASMRuntimeError(f"{rule} expects 1 argument", location=location, rewrite_rule=rule)
        a = args[0]
        if a.type == TYPE_INT:
            return TYPE_INT, self._expect_int(a, rule, location)
        if a.type == TYPE_FLT:
            return TYPE_FLT, self._expect_flt(a, rule, location)
        raise ASMRuntimeError(f"{rule} expects INT or FLT arguments", location=location, rewrite_rule=rule)

    def _add(self, _: "Interpreter", args: List[Value], __: List[Expression], ___: Environment, location: SourceLocation) -> Value:
        t, a, b = self._expect_num_pair(args, "ADD", location)
        return Value(t, a + b)

    def _sub(self, _: "Interpreter", args: List[Value], __: List[Expression], ___: Environment, location: SourceLocation) -> Value:
        t, a, b = self._expect_num_pair(args, "SUB", location)
        return Value(t, a - b)

    def _mul(self, _: "Interpreter", args: List[Value], __: List[Expression], ___: Environment, location: SourceLocation) -> Value:
        t, a, b = self._expect_num_pair(args, "MUL", location)
        return Value(t, a * b)

    def _div(self, _: "Interpreter", args: List[Value], __: List[Expression], ___: Environment, location: SourceLocation) -> Value:
        t, a, b = self._expect_num_pair(args, "DIV", location)
        if t == TYPE_INT:
            return Value(TYPE_INT, self._safe_div(a, b))
        if b == 0.0:
            raise ASMRuntimeError("Division by zero", rewrite_rule="DIV", location=location)
        return Value(TYPE_FLT, a / b)

    def _mod(self, _: "Interpreter", args: List[Value], __: List[Expression], ___: Environment, location: SourceLocation) -> Value:
        t, a, b = self._expect_num_pair(args, "MOD", location)
        if t == TYPE_INT:
            return Value(TYPE_INT, self._safe_mod(a, b))
        if b == 0.0:
            raise ASMRuntimeError("Division by zero", rewrite_rule="MOD", location=location)
        return Value(TYPE_FLT, a % abs(b))

    def _pow(self, _: "Interpreter", args: List[Value], __: List[Expression], ___: Environment, location: SourceLocation) -> Value:
        t, a, b = self._expect_num_pair(args, "POW", location)
        if t == TYPE_INT:
            return Value(TYPE_INT, self._safe_pow(a, b))
        return Value(TYPE_FLT, pow(a, b))

    def _root(self, _: "Interpreter", args: List[Value], __: List[Expression], ___: Environment, location: SourceLocation) -> Value:
        t, x, n = self._expect_num_pair(args, "ROOT", location)
        # _expect_num_pair enforces both args are same numeric type (INT or FLT)
        # Common checks
        if (t == TYPE_INT and n == 0) or (t == TYPE_FLT and n == 0.0):
            raise ASMRuntimeError("ROOT exponent must be non-zero", rewrite_rule="ROOT", location=location)

        if t == TYPE_INT:
            # INT path: n and x are ints
            # Allow negative x. For negative n, an integer result only exists in
            # trivial cases (|x| == 1). Disallow otherwise to avoid non-integer results.
            if n < 0:
                if x == 0:
                    raise ASMRuntimeError("Division by zero", rewrite_rule="ROOT", location=location)
                if abs(x) != 1:
                    raise ASMRuntimeError("Negative ROOT exponent yields non-integer result", rewrite_rule="ROOT", location=location)
                # 1**anything == 1 ; (-1)**odd == -1 ; reciprocal of ±1 is itself
                return Value(TYPE_INT, x)

            # n > 0: compute integer nth root (floor root for x>=0, sign for odd n)
            k = n
            if k == 1:
                return Value(TYPE_INT, x)
            if x >= 0:
                lo = 0
                hi = 1
                while pow(hi, k) <= x:
                    hi <<= 1
                while lo + 1 < hi:
                    mid = (lo + hi) // 2
                    if pow(mid, k) <= x:
                        lo = mid
                    else:
                        hi = mid
                return Value(TYPE_INT, lo)
            # x < 0: only odd roots yield integer results
            if k % 2 == 0:
                raise ASMRuntimeError("Even root of negative integer", rewrite_rule="ROOT", location=location)
            ax = -x
            lo = 0
            hi = 1
            while pow(hi, k) <= ax:
                hi <<= 1
            while lo + 1 < hi:
                mid = (lo + hi) // 2
                if pow(mid, k) <= ax:
                    lo = mid
                else:
                    hi = mid
            return Value(TYPE_INT, -lo)

        # FLT path
        # n may be negative; handle sign rules for negative base
        if x == 0.0 and n < 0.0:
            raise ASMRuntimeError("Division by zero", rewrite_rule="ROOT", location=location)
        if x < 0.0:
            # Negative base: allow only when n is integer and odd
            if not float(n).is_integer() or int(n) % 2 == 0:
                raise ASMRuntimeError("ROOT of negative float requires odd integer root", rewrite_rule="ROOT", location=location)
            # Compute signed result; pow handles fractional positive n
            sign = -1.0
            return Value(TYPE_FLT, sign * pow(abs(x), 1.0 / n))
        # general positive base (or zero with non-negative n)
        return Value(TYPE_FLT, pow(x, 1.0 / n))

    def _neg(self, _: "Interpreter", args: List[Value], __: List[Expression], ___: Environment, location: SourceLocation) -> Value:
        t, a = self._expect_num_unary(args, "NEG", location)
        return Value(t, -a)

    def _abs(self, _: "Interpreter", args: List[Value], __: List[Expression], ___: Environment, location: SourceLocation) -> Value:
        t, a = self._expect_num_unary(args, "ABS", location)
        return Value(t, abs(a))

    def _gcd(self, _: "Interpreter", args: List[Value], __: List[Expression], ___: Environment, location: SourceLocation) -> Value:
        t, a, b = self._expect_num_pair(args, "GCD", location)
        if t == TYPE_INT:
            return Value(TYPE_INT, math.gcd(a, b))
        # For floats, only accept integer-valued inputs.
        if not float(a).is_integer() or not float(b).is_integer():
            raise ASMRuntimeError("GCD expects integer-valued floats", location=location, rewrite_rule="GCD")
        return Value(TYPE_FLT, float(math.gcd(int(a), int(b))))

    def _lcm_num(self, _: "Interpreter", args: List[Value], __: List[Expression], ___: Environment, location: SourceLocation) -> Value:
        t, a, b = self._expect_num_pair(args, "LCM", location)
        if t == TYPE_INT:
            return Value(TYPE_INT, self._lcm(a, b))
        if not float(a).is_integer() or not float(b).is_integer():
            raise ASMRuntimeError("LCM expects integer-valued floats", location=location, rewrite_rule="LCM")
        ia, ib = int(a), int(b)
        if ia == 0 or ib == 0:
            return Value(TYPE_FLT, 0.0)
        return Value(TYPE_FLT, float(abs(ia * ib) // math.gcd(ia, ib)))

    def _gt(self, _: "Interpreter", args: List[Value], __: List[Expression], ___: Environment, location: SourceLocation) -> Value:
        _, a, b = self._expect_num_pair(args, "GT", location)
        return Value(TYPE_INT, 1 if a > b else 0)

    def _lt(self, _: "Interpreter", args: List[Value], __: List[Expression], ___: Environment, location: SourceLocation) -> Value:
        _, a, b = self._expect_num_pair(args, "LT", location)
        return Value(TYPE_INT, 1 if a < b else 0)

    def _gte(self, _: "Interpreter", args: List[Value], __: List[Expression], ___: Environment, location: SourceLocation) -> Value:
        _, a, b = self._expect_num_pair(args, "GTE", location)
        return Value(TYPE_INT, 1 if a >= b else 0)

    def _lte(self, _: "Interpreter", args: List[Value], __: List[Expression], ___: Environment, location: SourceLocation) -> Value:
        _, a, b = self._expect_num_pair(args, "LTE", location)
        return Value(TYPE_INT, 1 if a <= b else 0)

    def _log(self, _: "Interpreter", args: List[Value], __: List[Expression], ___: Environment, location: SourceLocation) -> Value:
        t, a = self._expect_num_unary(args, "LOG", location)
        if t == TYPE_INT:
            return Value(TYPE_INT, self._safe_log(a))
        if a <= 0.0:
            raise ASMRuntimeError("LOG argument must be > 0", rewrite_rule="LOG", location=location)
        return Value(TYPE_FLT, float(math.floor(math.log2(a))))

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

    def _as_bool_value(self, interpreter: "Interpreter", value: Value, location: SourceLocation) -> int:
        return 1 if interpreter._condition_int(value, location) != 0 else 0

    def _condition_from_value(self, interpreter: "Interpreter", value: Value, location: SourceLocation) -> int:
        return interpreter._condition_int(value, location)

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
    def _sum(self, _: "Interpreter", values: List[Value], location: SourceLocation) -> Value:
        if not values:
            raise ASMRuntimeError("SUM requires at least one argument", rewrite_rule="SUM")
        first_type = values[0].type
        if first_type == TYPE_INT:
            ints = [self._expect_int(v, "SUM", location) for v in values]
            return Value(TYPE_INT, sum(ints))
        if first_type == TYPE_FLT:
            flts = [self._expect_flt(v, "SUM", location) for v in values]
            return Value(TYPE_FLT, float(sum(flts)))
        raise ASMRuntimeError("SUM expects INT or FLT arguments", location=location, rewrite_rule="SUM")

    def _prod(self, _: "Interpreter", values: List[Value], location: SourceLocation) -> Value:
        if not values:
            raise ASMRuntimeError("PROD requires at least one argument", rewrite_rule="PROD")
        first_type = values[0].type
        if first_type == TYPE_INT:
            ints = [self._expect_int(v, "PROD", location) for v in values]
            result = 1
            for val in ints:
                result *= val
            return Value(TYPE_INT, result)
        if first_type == TYPE_FLT:
            flts = [self._expect_flt(v, "PROD", location) for v in values]
            result_f = 1.0
            for val in flts:
                result_f *= val
            return Value(TYPE_FLT, float(result_f))
        raise ASMRuntimeError("PROD expects INT or FLT arguments", location=location, rewrite_rule="PROD")

    def _max(self, _: "Interpreter", values: List[Value], location: SourceLocation) -> Value:
        if not values:
            raise ASMRuntimeError("MAX requires at least one argument", rewrite_rule="MAX")
        first_type = values[0].type
        if first_type == TYPE_TNS:
            raise ASMRuntimeError("MAX cannot operate on tensors", rewrite_rule="MAX", location=location)
        if any(v.type != first_type for v in values):
            raise ASMRuntimeError("MAX cannot mix values of different types", rewrite_rule="MAX", location=location)
        if first_type == TYPE_INT:
            ints = [self._expect_int(v, "MAX", location) for v in values]
            return Value(TYPE_INT, max(ints))
        if first_type == TYPE_FLT:
            flts = [self._expect_flt(v, "MAX", location) for v in values]
            return Value(TYPE_FLT, float(max(flts)))
        strs = [self._expect_str(v, "MAX", location) for v in values]
        longest = max(strs, key=len)
        return Value(TYPE_STR, longest)

    def _min(self, _: "Interpreter", values: List[Value], location: SourceLocation) -> Value:
        if not values:
            raise ASMRuntimeError("MIN requires at least one argument", rewrite_rule="MIN")
        first_type = values[0].type
        if first_type == TYPE_TNS:
            raise ASMRuntimeError("MIN cannot operate on tensors", rewrite_rule="MIN", location=location)
        if any(v.type != first_type for v in values):
            raise ASMRuntimeError("MIN cannot mix values of different types", rewrite_rule="MIN", location=location)
        if first_type == TYPE_INT:
            ints = [self._expect_int(v, "MIN", location) for v in values]
            return Value(TYPE_INT, min(ints))
        if first_type == TYPE_FLT:
            flts = [self._expect_flt(v, "MIN", location) for v in values]
            return Value(TYPE_FLT, float(min(flts)))
        strs = [self._expect_str(v, "MIN", location) for v in values]
        shortest = min(strs, key=len)
        return Value(TYPE_STR, shortest)

    def _any(self, interpreter: "Interpreter", values: List[Value], location: SourceLocation) -> Value:
        return Value(TYPE_INT, 1 if any(interpreter._condition_int(v, location) != 0 for v in values) else 0)

    def _all(self, interpreter: "Interpreter", values: List[Value], location: SourceLocation) -> Value:
        return Value(TYPE_INT, 1 if all(interpreter._condition_int(v, location) != 0 for v in values) else 0)

    def _len(self, _: "Interpreter", values: List[Value], __: SourceLocation) -> Value:
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

    def _join(self, _: "Interpreter", values: List[Value], location: SourceLocation) -> Value:
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
    def _and(self, interpreter: "Interpreter", args: List[Value], __: List[Expression], ___: Environment, location: SourceLocation) -> Value:
        a, b = args
        return Value(TYPE_INT, 1 if (self._as_bool_value(interpreter, a, location) and self._as_bool_value(interpreter, b, location)) else 0)

    def _or(self, interpreter: "Interpreter", args: List[Value], __: List[Expression], ___: Environment, location: SourceLocation) -> Value:
        a, b = args
        return Value(TYPE_INT, 1 if (self._as_bool_value(interpreter, a, location) or self._as_bool_value(interpreter, b, location)) else 0)

    def _xor(self, interpreter: "Interpreter", args: List[Value], __: List[Expression], ___: Environment, location: SourceLocation) -> Value:
        a, b = args
        return Value(TYPE_INT, 1 if (self._as_bool_value(interpreter, a, location) ^ self._as_bool_value(interpreter, b, location)) else 0)

    def _not(self, interpreter: "Interpreter", args: List[Value], __: List[Expression], ___: Environment, loc: SourceLocation) -> Value:
        return Value(TYPE_INT, 1 if self._as_bool_value(interpreter, args[0], loc) == 0 else 0)

    def _bool(self, interpreter: "Interpreter", args: List[Value], __: List[Expression], ___: Environment, loc: SourceLocation) -> Value:
        # BOOL(ANY: item):INT -> truthiness of item (INT: nonzero, STR: non-empty, TNS: any true element)
        return Value(TYPE_INT, 1 if self._as_bool_value(interpreter, args[0], loc) != 0 else 0)

    def _eq(self, interpreter: "Interpreter", args: List[Value], __: List[Expression], ___: Environment, ___loc: SourceLocation) -> Value:
        a, b = args
        return Value(TYPE_INT, 1 if interpreter._values_equal(a, b) else 0)

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
        if val.type == TYPE_FLT:
            # Explicit conversion only; truncate toward zero.
            return Value(TYPE_INT, int(float(val.value)))
        text = self._expect_str(val, "INT", location)
        if text == "":
            return Value(TYPE_INT, 0)
        if set(text).issubset({"0", "1"}):
            return Value(TYPE_INT, int(text, 2))
        return Value(TYPE_INT, 1)

    def _flt_op(self, _: "Interpreter", args: List[Value], __: List[Expression], ___: Environment, location: SourceLocation) -> Value:
        val = args[0]
        if val.type == TYPE_FLT:
            return val
        if val.type == TYPE_INT:
            return Value(TYPE_FLT, float(self._expect_int(val, "FLT", location)))
        text = self._expect_str(val, "FLT", location)
        if text == "":
            return Value(TYPE_FLT, 0.0)
        # Accept binary int strings and binary float strings.
        neg = text.startswith("-")
        core = text[1:] if neg else text
        if "." in core:
            left, right = core.split(".", 1)
            if left and right and set(left + right).issubset({"0", "1"}):
                num = (int(left, 2) << len(right)) + int(right, 2)
                den = 1 << len(right)
                out = float(num) / float(den)
                return Value(TYPE_FLT, -out if neg else out)
        if set(core).issubset({"0", "1"}):
            out = float(int(core, 2))
            return Value(TYPE_FLT, -out if neg else out)
        return Value(TYPE_FLT, 1.0)

    def _isflt(
        self,
        interpreter: "Interpreter",
        args: List[Value],
        arg_nodes: List[Expression],
        env: Environment,
        location: SourceLocation,
    ) -> Value:
        if not arg_nodes or not isinstance(arg_nodes[0], Identifier):
            raise ASMRuntimeError("ISFLT requires an identifier argument", location=location, rewrite_rule="ISFLT")
        name = arg_nodes[0].name
        if not env.has(name):
            return Value(TYPE_INT, 0)
        try:
            val = env.get(name)
        except ASMRuntimeError as err:
            err.location = location
            raise
        return Value(TYPE_INT, 1 if val.type == TYPE_FLT else 0)

    def _round(
        self,
        interpreter: "Interpreter",
        args: List[Value],
        __: List[Expression],
        ___: Environment,
        location: SourceLocation,
    ) -> Value:
        # ROUND(FLT: float, STR: mode = "floor", INT: ndigits = 0):FLT
        x = self._expect_flt(args[0], "ROUND", location)
        mode = "floor"
        ndigits = 0
        if len(args) >= 2:
            second = args[1]
            if second.type == TYPE_INT:
                ndigits = self._expect_int(second, "ROUND", location)
            else:
                mode = self._expect_str(second, "ROUND", location)
        if len(args) >= 3:
            mode = self._expect_str(args[1], "ROUND", location)
            ndigits = self._expect_int(args[2], "ROUND", location)
        mode_norm = mode.strip().lower()
        if mode_norm == "ceil":
            mode_norm = "ceiling"
        if mode_norm == "half-up":
            mode_norm = "logical"

        from fractions import Fraction

        frac = Fraction(*float(x).as_integer_ratio())
        if ndigits >= 0:
            scale = 1 << ndigits
            scaled = frac * scale
        else:
            scale = 1 << (-ndigits)
            scaled = frac / scale

        # scaled is a rational; round it to an integer per mode.
        n = scaled.numerator
        d = scaled.denominator
        if d == 0:
            raise ASMRuntimeError("ROUND internal error", location=location, rewrite_rule="ROUND")

        def _floor_div(a: int, b: int) -> int:
            return a // b

        def _ceil_div(a: int, b: int) -> int:
            return -((-a) // b)

        if mode_norm == "floor":
            q = _floor_div(n, d)
        elif mode_norm == "ceiling":
            q = _ceil_div(n, d)
        elif mode_norm == "zero":
            q = int(n / d)
        elif mode_norm == "logical":
            # Half-up: ties go away from zero.
            if n >= 0:
                q = _floor_div(2 * n + d, 2 * d)
            else:
                q = _ceil_div(2 * n - d, 2 * d)
        else:
            raise ASMRuntimeError(
                "ROUND mode must be one of floor, ceiling/ceil, zero, logical/half-up",
                location=location,
                rewrite_rule="ROUND",
            )

        if ndigits >= 0:
            out = Fraction(q, scale)
        else:
            out = Fraction(q * scale, 1)
        return Value(TYPE_FLT, float(out))

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
        pre_function_keys = set(interpreter.functions.keys())
        # Accept either IMPORT(module) or IMPORT(module, alias)
        if len(arg_nodes) not in (1, 2) or not isinstance(arg_nodes[0], Identifier):
            raise ASMRuntimeError("IMPORT expects module name identifier", location=location, rewrite_rule="IMPORT")

        module_name = arg_nodes[0].name
        if len(arg_nodes) == 2:
            if not isinstance(arg_nodes[1], Identifier):
                raise ASMRuntimeError("IMPORT alias must be an identifier", location=location, rewrite_rule="IMPORT")
            export_prefix = arg_nodes[1].name
        else:
            export_prefix = module_name
        # If module was already imported earlier in this interpreter instance,
        # reuse the same Environment and function objects so all importers
        # observe the same namespace/instance.
        if module_name in interpreter.module_cache:
            cached_env = interpreter.module_cache[module_name]
            # Ensure module functions are registered in the interpreter.functions map
            for fn in interpreter.module_functions.get(module_name, []):
                if fn.name not in interpreter.functions:
                    interpreter.functions[fn.name] = fn
                # Also register alias-qualified function names when an alias was requested
                if export_prefix != module_name:
                    # fn.name is module_name.func; produce alias.func
                    parts = fn.name.split(".", 1)
                    if len(parts) == 2:
                        _, unqualified = parts
                        alias_name = f"{export_prefix}.{unqualified}"
                        if alias_name not in interpreter.functions:
                            created = Function(
                                name=alias_name,
                                params=fn.params,
                                return_type=fn.return_type,
                                body=fn.body,
                                closure=cached_env,
                            )
                            interpreter.functions[alias_name] = created

            for k, v in cached_env.values.items():
                dotted = f"{export_prefix}.{k}"
                env.set(dotted, v, declared_type=v.type)

            if set(interpreter.functions.keys()) != pre_function_keys:
                interpreter._mark_functions_changed()
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

        # --- IMPORT-time extension loading ---
        # When importing a module, attempt to load any companion extensions so
        # their operators are available immediately. Look for a companion
        # pointer file (<module>.asmxt) next to the module file (or the copy
        # in the interpreter's lib/), and also check the interpreter's built-in
        # ext/ directory for an <module>.py extension module.
        try:
            import extensions as _extmod

            # If a companion .asmxt exists alongside the module, load listed extensions.
            companion_asmxt = os.path.splitext(module_path)[0] + ".asmxt"
            if os.path.exists(companion_asmxt):
                try:
                    asm_paths = _extmod.read_asmx(companion_asmxt)
                    for p in _extmod.gather_extension_paths(asm_paths):
                        mod = _extmod.load_extension_module(p)
                        api_version = getattr(mod, "ASM_LANG_EXTENSION_API_VERSION", _extmod.EXTENSION_API_VERSION)
                        if api_version != _extmod.EXTENSION_API_VERSION:
                            raise ASMExtensionError(
                                f"Extension {p} requires API {api_version}, host supports {_extmod.EXTENSION_API_VERSION}"
                            )
                        register = getattr(mod, "asm_lang_register", None)
                        if register is None or not callable(register):
                            raise ASMExtensionError(f"Extension {p} must define callable asm_lang_register(ext)")
                        ext_name = getattr(mod, "ASM_LANG_EXTENSION_NAME", os.path.splitext(os.path.basename(p))[0])
                        ext_asmodule = bool(getattr(mod, "ASM_LANG_EXTENSION_ASMODULE", False))
                        ext_api = _extmod.ExtensionAPI(services=interpreter.services, ext_name=str(ext_name), asmodule=ext_asmodule)
                        before = len(interpreter.services.operators)
                        register(ext_api)
                        # Register any newly-added operators into the running interpreter
                        for name, min_args, max_args, impl, _doc in interpreter.services.operators[before:]:
                            # Avoid duplicate registration when the same extension is loaded again
                            if name in interpreter.builtins.table:
                                continue
                            interpreter.builtins.register_extension_operator(
                                name=name,
                                min_args=min_args,
                                max_args=max_args,
                                impl=impl,
                            )
                except ASMExtensionError as exc:
                    raise ASMExtensionError(f"Failed to load extensions from {companion_asmxt}: {exc}") from exc

            # Next, check for a single-file built-in extension named <module>.py
            builtin = _extmod._resolve_in_builtin_ext(f"{module_name}.py")
            if builtin is not None and os.path.exists(builtin):
                mod = _extmod.load_extension_module(builtin)
                api_version = getattr(mod, "ASM_LANG_EXTENSION_API_VERSION", _extmod.EXTENSION_API_VERSION)
                if api_version != _extmod.EXTENSION_API_VERSION:
                    raise ASMExtensionError(
                        f"Extension {builtin} requires API {api_version}, host supports {_extmod.EXTENSION_API_VERSION}"
                    )
                register = getattr(mod, "asm_lang_register", None)
                if register is not None and callable(register):
                    ext_name = getattr(mod, "ASM_LANG_EXTENSION_NAME", os.path.splitext(os.path.basename(builtin))[0])
                    ext_asmodule = bool(getattr(mod, "ASM_LANG_EXTENSION_ASMODULE", False))
                    ext_api = _extmod.ExtensionAPI(services=interpreter.services, ext_name=str(ext_name), asmodule=ext_asmodule)
                    before = len(interpreter.services.operators)
                    register(ext_api)
                    for name, min_args, max_args, impl, _doc in interpreter.services.operators[before:]:
                        if name in interpreter.builtins.table:
                            continue
                        interpreter.builtins.register_extension_operator(
                            name=name,
                            min_args=min_args,
                            max_args=max_args,
                            impl=impl,
                        )
        except ASMExtensionError as exc:
            raise ASMRuntimeError(str(exc), location=location, rewrite_rule="IMPORT")
        except Exception:
            # Non-fatal: do not prevent importing the asm module if extension
            # loading fails unexpectedly; convert to runtime error only when
            # it's an ASMExtensionError above.
            pass

        lexer = Lexer(source_text, module_path)
        tokens = lexer.tokenize()
        parser = Parser(tokens, module_path, source_text.splitlines(), type_names=interpreter.type_registry.names())
        program = parser.parse()

        module_env = Environment()
        prev_functions = dict(interpreter.functions)
        try:
            interpreter._execute_block(program.statements, module_env)
        except Exception as exc:
            # Restore interpreter function table on error and convert
            # unexpected Python exceptions into ASMRuntimeError so they
            # are reported using the language's traceback machinery.
            interpreter.functions = prev_functions
            if isinstance(exc, ASMRuntimeError):
                raise
            raise ASMRuntimeError(f"Import failed: {exc}", location=location, rewrite_rule="IMPORT")

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

        # If the caller requested an alias different from the module name,
        # also register alias-qualified function entries pointing to the
        # same function bodies so callers can invoke e.g. ALIAS.F().
        if export_prefix != module_name:
            for fn in registered_functions:
                parts = fn.name.split(".", 1)
                if len(parts) == 2:
                    _, unqualified = parts
                    alias_name = f"{export_prefix}.{unqualified}"
                    if alias_name not in interpreter.functions:
                        alias_fn = Function(
                            name=alias_name,
                            params=fn.params,
                            return_type=fn.return_type,
                            body=fn.body,
                            closure=module_env,
                        )
                        interpreter.functions[alias_name] = alias_fn

        # Store module env and functions into the interpreter cache for reuse
        interpreter.module_cache[module_name] = module_env
        interpreter.module_functions[module_name] = registered_functions

        if set(interpreter.functions.keys()) != pre_function_keys:
            interpreter._mark_functions_changed()

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
        parser = Parser(tokens, run_filename, source_text.splitlines(), type_names=interpreter.type_registry.names())
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
                spec = interpreter.type_registry.get_optional(arg.type)
                if spec is not None and spec.printable:
                    ctx = TypeContext(interpreter=interpreter, location=loc)
                    rendered.append(spec.to_str(ctx, arg))
                else:
                    raise ASMRuntimeError(
                        "PRINT accepts INT or STR arguments",
                        location=loc,
                        rewrite_rule="PRINT",
                    )
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
        cond = self._condition_from_value(interpreter, args[0], location) if args else 0
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

    def _type(
        self,
        interpreter: "Interpreter",
        args: List[Value],
        arg_nodes: List[Expression],
        env: Environment,
        location: SourceLocation,
    ) -> Value:
        # TYPE(ANY: obj):STR -> return the runtime type name of the value as STR
        if len(args) != 1:
            raise ASMRuntimeError("TYPE expects one argument", location=location, rewrite_rule="TYPE")
        val = args[0]
        # Return the type string as STR; preserve extension type names if present
        return Value(TYPE_STR, val.type)

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
        out = np.empty(x.data.size, dtype=object)
        i = 0
        expect_int = self._expect_int
        for a, b in zip(x.data.flat, y.data.flat):
            out[i] = Value(TYPE_INT, op(expect_int(a, rule, location), expect_int(b, rule, location)))
            i += 1
        return Tensor(shape=list(x.shape), data=out)

    def _map_tensor_numeric_binary(
        self,
        x: Tensor,
        y: Tensor,
        rule: str,
        location: SourceLocation,
        op_int: Callable[[int, int], int],
        op_flt: Callable[[float, float], float],
    ) -> Tensor:
        if x.shape != y.shape:
            raise ASMRuntimeError(f"{rule} requires tensors with identical shapes", location=location, rewrite_rule=rule)
        if x.data.size == 0:
            # Shapes are validated elsewhere to be positive, but keep safe.
            return Tensor(shape=list(x.shape), data=np.array([], dtype=object))
        first_type = x.data.flat[0].type
        if first_type not in (TYPE_INT, TYPE_FLT):
            raise ASMRuntimeError(f"{rule} expects INT or FLT tensor elements", location=location, rewrite_rule=rule)
        if any(v.type != first_type for v in x.data.flat) or any(v.type != first_type for v in y.data.flat):
            raise ASMRuntimeError(f"{rule} cannot mix INT and FLT", location=location, rewrite_rule=rule)
        if first_type == TYPE_INT:
            out = np.empty(x.data.size, dtype=object)
            i = 0
            expect_int = self._expect_int
            for a, b in zip(x.data.flat, y.data.flat):
                out[i] = Value(TYPE_INT, op_int(expect_int(a, rule, location), expect_int(b, rule, location)))
                i += 1
            return Tensor(shape=list(x.shape), data=out)

        out = np.empty(x.data.size, dtype=object)
        i = 0
        for a, b in zip(x.data.flat, y.data.flat):
            out[i] = Value(TYPE_FLT, op_flt(float(a.value), float(b.value)))
            i += 1
        return Tensor(shape=list(x.shape), data=out)

    def _map_tensor_int_scalar(self, tensor: Tensor, scalar: int, rule: str, location: SourceLocation, op: Callable[[int, int], int]) -> Tensor:
        out = np.empty(tensor.data.size, dtype=object)
        i = 0
        expect_int = self._expect_int
        for entry in tensor.data.flat:
            out[i] = Value(TYPE_INT, op(expect_int(entry, rule, location), scalar))
            i += 1
        return Tensor(shape=list(tensor.shape), data=out)

    def _map_tensor_numeric_scalar(
        self,
        tensor: Tensor,
        scalar: Value,
        rule: str,
        location: SourceLocation,
        op_int: Callable[[int, int], int],
        op_flt: Callable[[float, float], float],
    ) -> Tensor:
        if tensor.data.size == 0:
            return Tensor(shape=list(tensor.shape), data=np.array([], dtype=object))
        first_type = tensor.data.flat[0].type
        if first_type not in (TYPE_INT, TYPE_FLT):
            raise ASMRuntimeError(f"{rule} expects INT or FLT tensor elements", location=location, rewrite_rule=rule)
        if scalar.type != first_type:
            raise ASMRuntimeError(f"{rule} cannot mix INT and FLT", location=location, rewrite_rule=rule)
        if any(v.type != first_type for v in tensor.data.flat):
            raise ASMRuntimeError(f"{rule} tensor has mixed element types", location=location, rewrite_rule=rule)
        if first_type == TYPE_INT:
            sc = self._expect_int(scalar, rule, location)
            out = np.empty(tensor.data.size, dtype=object)
            i = 0
            expect_int = self._expect_int
            for entry in tensor.data.flat:
                out[i] = Value(TYPE_INT, op_int(expect_int(entry, rule, location), sc))
                i += 1
            return Tensor(shape=list(tensor.shape), data=out)
        scf = float(scalar.value)
        out = np.empty(tensor.data.size, dtype=object)
        i = 0
        for entry in tensor.data.flat:
            out[i] = Value(TYPE_FLT, op_flt(float(entry.value), scf))
            i += 1
        return Tensor(shape=list(tensor.shape), data=out)

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
        new_data = np.empty(tensor.data.size, dtype=object)
        for i in range(tensor.data.size):
            new_data[i] = Value(fill_value.type, fill_value.value)
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
        tensor = self._map_tensor_numeric_binary(x, y, "MADD", location, lambda a, b: a + b, lambda a, b: a + b)
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
        tensor = self._map_tensor_numeric_binary(x, y, "MSUB", location, lambda a, b: a - b, lambda a, b: a - b)
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
        tensor = self._map_tensor_numeric_binary(x, y, "MMUL", location, lambda a, b: a * b, lambda a, b: a * b)
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
        def _div_int(a: int, b: int) -> int:
            return self._safe_div(a, b)

        def _div_flt(a: float, b: float) -> float:
            if b == 0.0:
                raise ASMRuntimeError("Division by zero", rewrite_rule="MDIV", location=location)
            return a / b

        tensor = self._map_tensor_numeric_binary(x, y, "MDIV", location, _div_int, _div_flt)
        return Value(TYPE_TNS, tensor)

    def _msum(self, _: "Interpreter", values: List[Value], location: SourceLocation) -> Value:
        tensors = [self._expect_tns(v, "MSUM", location) for v in values]
        acc = tensors[0]
        for tensor in tensors[1:]:
            acc = self._map_tensor_numeric_binary(acc, tensor, "MSUM", location, lambda a, b: a + b, lambda a, b: a + b)
        return Value(TYPE_TNS, acc)

    def _mprod(self, _: "Interpreter", values: List[Value], location: SourceLocation) -> Value:
        tensors = [self._expect_tns(v, "MPROD", location) for v in values]
        acc = tensors[0]
        for tensor in tensors[1:]:
            acc = self._map_tensor_numeric_binary(acc, tensor, "MPROD", location, lambda a, b: a * b, lambda a, b: a * b)
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
        scalar = args[1]
        result = self._map_tensor_numeric_scalar(tensor, scalar, "TADD", location, lambda a, b: a + b, lambda a, b: a + b)
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
        scalar = args[1]
        result = self._map_tensor_numeric_scalar(tensor, scalar, "TSUB", location, lambda a, b: a - b, lambda a, b: a - b)
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
        scalar = args[1]
        result = self._map_tensor_numeric_scalar(tensor, scalar, "TMUL", location, lambda a, b: a * b, lambda a, b: a * b)
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
        scalar = args[1]

        def _div_int(val: int, sc: int) -> int:
            return self._safe_div(val, sc)

        def _div_flt(val: float, sc: float) -> float:
            if sc == 0.0:
                raise ASMRuntimeError("Division by zero", rewrite_rule="TDIV", location=location)
            return val / sc

        result = self._map_tensor_numeric_scalar(tensor, scalar, "TDIV", location, _div_int, _div_flt)
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
        scalar = args[1]

        def _pow_int(a: int, b: int) -> int:
            return self._safe_pow(a, b)

        def _pow_flt(a: float, b: float) -> float:
            return pow(a, b)

        result = self._map_tensor_numeric_scalar(tensor, scalar, "TPOW", location, _pow_int, _pow_flt)
        return Value(TYPE_TNS, result)

    def _flip(
        self,
        interpreter: "Interpreter",
        args: List[Value],
        __: List[Expression],
        ___: Environment,
        location: SourceLocation,
    ) -> Value:
        """FLIP(INT|STR: obj):INT|STR
        - INT: reverse the binary-digit spelling of the absolute value and preserve sign
        - STR: reverse the character sequence
        """
        val = args[0]
        if val.type == TYPE_INT:
            n = self._expect_int(val, "FLIP", location)
            neg = n < 0
            a = abs(n)
            bits = format(a, "b")
            rev = bits[::-1]
            result = int(rev, 2) if rev != "" else 0
            return Value(TYPE_INT, -result if neg else result)
        if val.type == TYPE_STR:
            s = self._expect_str(val, "FLIP", location)
            return Value(TYPE_STR, s[::-1])
        raise ASMRuntimeError("FLIP expects INT or STR", location=location, rewrite_rule="FLIP")

    def _tflip(
        self,
        interpreter: "Interpreter",
        args: List[Value],
        __: List[Expression],
        ___: Environment,
        location: SourceLocation,
    ) -> Value:
        """TFLIP(TNS: obj, INT: dim):TNS
        Return a new tensor with dimension `dim` reversed (1-based).
        """
        tensor = self._expect_tns(args[0], "TFLIP", location)
        dim = self._expect_int(args[1], "TFLIP", location)
        if dim <= 0 or dim > len(tensor.shape):
            raise ASMRuntimeError("TFLIP dimension out of range", location=location, rewrite_rule="TFLIP")
        axis = dim - 1
        # reshape the flat data into the tensor shape, flip along axis, then flatten
        reshaped = tensor.data.reshape(tuple(tensor.shape))
        flipped = np.flip(reshaped, axis=axis)
        # ravel().copy() avoids constructing a Python list and preserves order.
        new_data = flipped.ravel().copy()
        return Value(TYPE_TNS, Tensor(shape=list(tensor.shape), data=new_data))

    def _convolve(
        self,
        interpreter: "Interpreter",
        args: List[Value],
        __: List[Expression],
        ___: Environment,
        location: SourceLocation,
    ) -> Value:
        """CONVOLVE(TNS: x, TNS: kernel):TNS

        N-dimensional discrete convolution with clamped (replicate) boundary.

        - `x` and `kernel` must have the same rank.
        - Each `kernel` dimension length must be odd (so it has a well-defined center).
        - Tensor elements must be uniformly INT or uniformly FLT within each tensor.
        - If both tensors are INT -> output elements are INT.
          Otherwise output elements are FLT (INT/FLT mixing is allowed and produces FLT).

        The output shape equals the input `x` shape.
        """

        x = self._expect_tns(args[0], "CONVOLVE", location)
        kernel = self._expect_tns(args[1], "CONVOLVE", location)

        if len(x.shape) != len(kernel.shape):
            raise ASMRuntimeError(
                "CONVOLVE requires input and kernel tensors with the same rank",
                location=location,
                rewrite_rule="CONVOLVE",
            )
        if any((d % 2) == 0 for d in kernel.shape):
            raise ASMRuntimeError(
                "CONVOLVE requires odd kernel dimensions",
                location=location,
                rewrite_rule="CONVOLVE",
            )

        def _uniform_numeric_type(t: Tensor, which: str) -> str:
            if t.data.size == 0:
                raise ASMRuntimeError(
                    f"CONVOLVE does not support empty {which} tensors",
                    location=location,
                    rewrite_rule="CONVOLVE",
                )
            first = next(iter(t.data.flat)).type
            if first not in (TYPE_INT, TYPE_FLT):
                raise ASMRuntimeError(
                    "CONVOLVE expects INT or FLT tensor elements",
                    location=location,
                    rewrite_rule="CONVOLVE",
                )
            if any(v.type != first for v in t.data.flat):
                raise ASMRuntimeError(
                    "CONVOLVE does not allow mixed element types within a tensor",
                    location=location,
                    rewrite_rule="CONVOLVE",
                )
            return first

        x_type = _uniform_numeric_type(x, "input")
        k_type = _uniform_numeric_type(kernel, "kernel")
        out_type = TYPE_INT if (x_type == TYPE_INT and k_type == TYPE_INT) else TYPE_FLT

        rank = len(x.shape)
        centers = [d // 2 for d in kernel.shape]  # 0-based

        # Fast path: use NumPy padding + sliding windows + vectorized multiply/sum.
        # This keeps exact boundary semantics (replicate) and true convolution (kernel flipped).
        try:
            sliding_window_view = np.lib.stride_tricks.sliding_window_view  # type: ignore[attr-defined]

            if out_type == TYPE_INT:
                x_int = np.fromiter((int(v.value) for v in x.data.flat), dtype=object, count=x.data.size).reshape(tuple(x.shape))
                k_int = np.fromiter((int(v.value) for v in kernel.data.flat), dtype=object, count=kernel.data.size).reshape(tuple(kernel.shape))
                k_flip = np.flip(k_int, axis=tuple(range(rank)))
                pad_width = [(c, c) for c in centers]
                padded = np.pad(x_int, pad_width=pad_width, mode="edge")
                windows = sliding_window_view(padded, tuple(kernel.shape))
                acc = (windows * k_flip).sum(axis=tuple(range(-rank, 0)))
                out = np.empty(x.data.size, dtype=object)
                i = 0
                for val in acc.flat:
                    out[i] = Value(TYPE_INT, val)
                    i += 1
                return Value(TYPE_TNS, Tensor(shape=list(x.shape), data=out))

            # FLT output: allow INT/FLT mixing by converting to float.
            def _as_float(v: Value) -> float:
                return float(v.value) if v.type == TYPE_FLT else float(int(v.value))

            x_flt = np.fromiter((_as_float(v) for v in x.data.flat), dtype=float, count=x.data.size).reshape(tuple(x.shape))
            k_flt = np.fromiter((_as_float(v) for v in kernel.data.flat), dtype=float, count=kernel.data.size).reshape(tuple(kernel.shape))
            k_flip = np.flip(k_flt, axis=tuple(range(rank)))
            pad_width = [(c, c) for c in centers]
            padded = np.pad(x_flt, pad_width=pad_width, mode="edge")
            windows = sliding_window_view(padded, tuple(kernel.shape))
            acc = (windows * k_flip).sum(axis=tuple(range(-rank, 0)))
            out = np.empty(x.data.size, dtype=object)
            i = 0
            for val in acc.flat:
                out[i] = Value(TYPE_FLT, float(val))
                i += 1
            return Value(TYPE_TNS, Tensor(shape=list(x.shape), data=out))

        except Exception:
            # Fallback to the reference implementation if sliding window view
            # isn't available or if NumPy raises in an edge case.
            x_arr = x.data.reshape(tuple(x.shape))
            k_arr = kernel.data.reshape(tuple(kernel.shape))
            out = np.empty(x.data.size, dtype=object)
            out_i = 0
            for out_pos in np.ndindex(*x.shape):
                acc_int = 0
                acc_flt = 0.0
                for k_pos in np.ndindex(*kernel.shape):
                    # Map kernel position to input position centered at out_pos.
                    in_pos: List[int] = []
                    for axis in range(rank):
                        offset = k_pos[axis] - centers[axis]
                        coord = out_pos[axis] + offset
                        if coord < 0:
                            coord = 0
                        elif coord >= x.shape[axis]:
                            coord = x.shape[axis] - 1
                        in_pos.append(coord)

                    # True convolution flips the kernel along every axis.
                    k_flip = tuple(kernel.shape[axis] - 1 - k_pos[axis] for axis in range(rank))
                    xv: Value = x_arr[tuple(in_pos)]
                    kv: Value = k_arr[k_flip]

                    if out_type == TYPE_INT:
                        acc_int += int(xv.value) * int(kv.value)
                    else:
                        ax = float(xv.value) if xv.type == TYPE_FLT else float(int(xv.value))
                        ak = float(kv.value) if kv.type == TYPE_FLT else float(int(kv.value))
                        acc_flt += ax * ak

                out[out_i] = Value(out_type, acc_int if out_type == TYPE_INT else acc_flt)
                out_i += 1

            return Value(TYPE_TNS, Tensor(shape=list(x.shape), data=out))

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
        services: Optional[RuntimeServices] = None,
        input_provider: Optional[Callable[[], str]] = None,
        output_sink: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.source = source
        self._source_lines = source.splitlines()
        normalized_filename = filename if filename == "<string>" else os.path.abspath(filename)
        self.filename = normalized_filename
        self.entry_filename = normalized_filename
        self.verbose = verbose
        self.services = services or build_default_services()
        self.type_registry: TypeRegistry = self.services.type_registry
        self.hook_registry: HookRegistry = self.services.hook_registry
        self.input_provider = input_provider or (lambda: input(">>> "))
        self.output_sink = output_sink or (lambda text: print(text))
        self.builtins = Builtins()

        # Install built-in types (INT/STR/TNS) into the registry with their
        # concrete runtime behavior. These names are reserved by default
        # services so extensions cannot redefine them.
        if not self.type_registry.has(TYPE_INT):
            self.type_registry.register(
                TypeSpec(
                    name=TYPE_INT,
                    printable=True,
                    condition_int=lambda ctx, v: int(getattr(v, "value", 0)),
                    to_str=lambda ctx, v: ("-" + format(-int(v.value), "b")) if int(v.value) < 0 else format(int(v.value), "b"),
                ),
                seal=True,
            )

        if not self.type_registry.has(TYPE_FLT):
            def _flt_to_str(ctx: TypeContext, v: Value) -> str:
                x = float(v.value)
                if x == 0.0:
                    return "0.0"
                neg = x < 0
                num, den = float(abs(x)).as_integer_ratio()
                # den is a power of two for IEEE754 floats.
                k = den.bit_length() - 1
                whole = num >> k
                rem = num & ((1 << k) - 1)
                whole_bits = format(whole, "b")
                frac_bits = format(rem, f"0{k}b") if k > 0 else "0"
                frac_bits = frac_bits.rstrip("0") or "0"
                return ("-" if neg else "") + whole_bits + "." + frac_bits

            self.type_registry.register(
                TypeSpec(
                    name=TYPE_FLT,
                    printable=True,
                    condition_int=lambda ctx, v: 0 if float(v.value) == 0.0 else 1,
                    to_str=_flt_to_str,
                ),
                seal=True,
            )
        if not self.type_registry.has(TYPE_STR):
            def _str_cond(ctx: TypeContext, v: Value) -> int:
                text = str(v.value)
                if text == "":
                    return 0
                if set(text).issubset({"0", "1"}):
                    return int(text, 2)
                return 1

            self.type_registry.register(
                TypeSpec(
                    name=TYPE_STR,
                    printable=True,
                    condition_int=_str_cond,
                    to_str=lambda ctx, v: str(v.value),
                ),
                seal=True,
            )
        if not self.type_registry.has(TYPE_TNS):
            self.type_registry.register(
                TypeSpec(
                    name=TYPE_TNS,
                    printable=False,  # preserve historical PRINT behavior
                    condition_int=lambda ctx, v: 1 if ctx.interpreter._tensor_truthy(v.value) else 0,
                    to_str=lambda ctx, v: "<tensor>",
                    equals=lambda ctx, a, b: ctx.interpreter._tensor_equal(a.value, b.value),
                ),
                seal=True,
            )

        # Attach extension-provided operators. These are appended to the builtins
        # table but cannot override existing operator names.
        for name, min_args, max_args, impl, _doc in self.services.operators:
            self.builtins.register_extension_operator(
                name=name,
                min_args=min_args,
                max_args=max_args,
                impl=impl,
            )

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

        # Cache for resolving unqualified function calls inside module frames.
        # Keyed by (frame_name, callee_name, functions_version).
        self._functions_version: int = 0
        self._call_resolution_cache: Dict[Tuple[str, str, int], Optional[str]] = {}

    def _mark_functions_changed(self) -> None:
        self._functions_version += 1
        self._call_resolution_cache.clear()

    def _resolve_user_function_name(self, *, frame_name: str, callee: str) -> Optional[str]:
        version = self._functions_version
        key = (frame_name, callee, version)
        cached = self._call_resolution_cache.get(key, None)
        # We cache positive and negative results; distinguish missing key from cached None
        if key in self._call_resolution_cache:
            return cached

        resolved: Optional[str] = None
        if callee in self.functions:
            resolved = callee
        else:
            dot = frame_name.find(".")
            if dot != -1:
                module_prefix = frame_name[:dot]
                candidate = f"{module_prefix}.{callee}"
                if candidate in self.functions:
                    resolved = candidate

        self._call_resolution_cache[key] = resolved
        return resolved

    # Convenience wrappers so extensions can call interpreter._expect_*
    # directly. Delegate to the builtins helpers which implement the
    # concrete checks and error reporting.
    def _expect_tns(self, value: "Value", rule: str, location: SourceLocation) -> "Tensor":
        return self.builtins._expect_tns(value, rule, location)

    def _expect_int(self, value: "Value", rule: str, location: SourceLocation) -> int:
        return self.builtins._expect_int(value, rule, location)

    def _expect_flt(self, value: "Value", rule: str, location: SourceLocation) -> float:
        return self.builtins._expect_flt(value, rule, location)

    def _expect_str(self, value: "Value", rule: str, location: SourceLocation) -> str:
        return self.builtins._expect_str(value, rule, location)
    def parse(self) -> Program:
        lexer = Lexer(self.source, self.filename)
        tokens = lexer.tokenize()
        parser = Parser(tokens, self.filename, self._source_lines, type_names=self.type_registry.names())
        return parser.parse()

    def run(self) -> None:
        program = self.parse()
        global_env = Environment()
        global_frame = self._new_frame("<top-level>", global_env, None)
        self.call_stack.append(global_frame)
        self._emit_event("program_start", self, program, global_env)
        try:
            self._execute_block(program.statements, global_env)
        except BreakSignal as bs:
            raise ASMRuntimeError(f"BREAK({bs.count}) escaped enclosing loops", rewrite_rule="BREAK")
        except ContinueSignal:
            raise ASMRuntimeError("CONTINUE used outside loop", rewrite_rule="CONTINUE")
        except ASMRuntimeError as error:
            self._emit_event("on_error", self, error)
            if self.logger.entries:
                error.step_index = self.logger.entries[-1].step_index
            raise
        except Exception as exc:
            self._emit_event("on_error", self, exc)
            # Convert unexpected Python-level exceptions into ASMRuntimeError
            # so callers (REPL/CLI) can format them using ASM-Lang tracebacks.
            loc = None
            if self.logger.entries:
                last = self.logger.entries[-1]
                loc = last.source_location
            wrapped = ASMRuntimeError(f"Internal interpreter error: {exc}", location=loc, rewrite_rule="internal")
            if self.logger.entries:
                wrapped.step_index = self.logger.entries[-1].step_index
            raise wrapped
        else:
            self._emit_event("program_end", self, 0)
            self.call_stack.pop()

    def _execute_block(self, statements: List[Statement], env: Environment) -> None:
        i = 0
        emit_event = self._emit_event
        log_step = self._log_step
        eval_expr = self._evaluate_expression
        execute_stmt = self._execute_statement

        frame: Frame = self.call_stack[-1]
        gotopoints: Dict[int, int] = frame.gotopoints
        while i < len(statements):
            statement = statements[i]
            emit_event("before_statement", self, statement, env)
            if isinstance(statement, GotopointStatement):
                log_step(rule=statement.__class__.__name__, location=statement.location)
                gid = eval_expr(statement.expression, env)
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
                emit_event("after_statement", self, statement, env)
                continue
            try:
                execute_stmt(statement, env)
            except JumpSignal as js:
                target = js.target
                key = (target.type, target.value)
                if key not in gotopoints:
                    raise ASMRuntimeError(
                        f"GOTO to undefined gotopoint '{target.value}'", location=statement.location, rewrite_rule="GOTO"
                    )
                i = gotopoints[key]
                emit_event("after_statement", self, statement, env)
                continue
            emit_event("after_statement", self, statement, env)
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
            eval_expr = self._evaluate_expression
            expect_int = self._expect_int
            indices: List[int] = []
            indices_append = indices.append
            for node in index_nodes:
                indices_append(expect_int(eval_expr(node, env), "ASSIGN", statement.location))
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
            # Function table mutation invalidates call resolution caches.
            self._mark_functions_changed()
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
                    spec = self.type_registry.get_optional(fn.return_type)
                    if spec is not None and spec.default_value is not None:
                        ctx = TypeContext(interpreter=self, location=statement.location)
                        value = spec.default_value(ctx)
                    else:
                        raise ASMRuntimeError(
                            f"Function {fn.name} must RETURN a {fn.return_type} value",
                            location=statement.location,
                            rewrite_rule="RETURN",
                        )
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
        eval_expr = self._evaluate_expression
        cond_int = self._condition_int
        if cond_int(eval_expr(statement.condition, env), statement.location) != 0:
            self._execute_block(statement.then_block.statements, env)
            return
        for branch in statement.elifs:
            if cond_int(eval_expr(branch.condition, env), branch.condition.location) != 0:
                self._execute_block(branch.block.statements, env)
                return
        if statement.else_block:
            self._execute_block(statement.else_block.statements, env)

    def _execute_while(self, statement: WhileStatement, env: Environment) -> None:
        eval_expr = self._evaluate_expression
        cond_int = self._condition_int
        while cond_int(eval_expr(statement.condition, env), statement.condition.location) != 0:
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
                    next_cond_val = eval_expr(statement.condition, env)
                    next_cond = cond_int(next_cond_val, statement.condition.location)
                except ASMRuntimeError:
                    # Propagate evaluation errors
                    raise
                if next_cond != 0:
                    # proceed to next iteration
                    continue
                # no next iteration -> behave like BREAK(1)
                return

    def _execute_for(self, statement: ForStatement, env: Environment) -> None:
        eval_expr = self._evaluate_expression
        target_val = eval_expr(statement.target_expr, env)
        target = self._expect_int(target_val, "FOR", statement.location)
        # FOR loops use 1-indexed counters (language-level). Initialize
        # counter to 1 and iterate while counter <= target so library code
        # that indexes tensors with the loop variable works correctly.
        env.set(statement.counter, Value(TYPE_INT, 1), declared_type=TYPE_INT)
        while self._expect_int(env.get(statement.counter), "FOR", statement.location) <= target:
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
                if current + 1 <= target:
                    env.set(statement.counter, Value(TYPE_INT, current + 1))
                    continue
                # no further iterations -> behave like BREAK(1)
                return
            env.set(statement.counter, Value(TYPE_INT, self._expect_int(env.get(statement.counter), "FOR", statement.location) + 1))

    def _condition_int(self, value: Value, location: Optional[SourceLocation]) -> int:
        # Fast-path the built-in scalar types; this avoids repeated registry
        # lookups and TypeContext allocations on tight control-flow loops.
        # Extension-defined types still go through the registry.
        vtype = value.type
        if vtype == TYPE_INT:
            return 0 if int(value.value) == 0 else 1
        if vtype == TYPE_FLT:
            return 0 if float(value.value) == 0.0 else 1
        if vtype == TYPE_STR:
            text = str(value.value)
            if text == "":
                return 0
            if set(text).issubset({"0", "1"}):
                return int(text, 2)
            return 1
        spec = self.type_registry.get_optional(value.type)
        if spec is None:
            raise ASMRuntimeError("Unsupported type in condition", location=location)
        ctx = TypeContext(interpreter=self, location=location)
        return int(spec.condition_int(ctx, value))

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
        cond_int = self._condition_int
        for item in tensor.data.flat:
            if cond_int(item, None) != 0:
                return True
        return False

    def _values_equal(self, left: Value, right: Value) -> bool:
        if left.type != right.type:
            return False
        if left.type == TYPE_TNS:
            assert isinstance(left.value, Tensor)
            assert isinstance(right.value, Tensor)
            return self._tensor_equal(left.value, right.value)
        # Fast-path the built-in scalar types.
        if left.type in (TYPE_INT, TYPE_FLT, TYPE_STR):
            return left.value == right.value

        spec = self.type_registry.get_optional(left.type)
        if spec is not None and spec.equals is not None:
            ctx = TypeContext(interpreter=self, location=None)
            return bool(spec.equals(ctx, left, right))
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
        shape = tensor.shape
        strides = tensor.strides
        for i, raw in enumerate(indices):
            dim_len = shape[i]
            idx = raw
            if idx == 0:
                raise ASMRuntimeError("Tensor indices are 1-indexed", location=location, rewrite_rule=rule)
            if idx < 0:
                idx = dim_len + idx + 1
            if idx <= 0 or idx > dim_len:
                raise ASMRuntimeError("Tensor index out of range", location=location, rewrite_rule=rule)
            offset += (idx - 1) * strides[i]
        return int(offset)

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
                # numpy.flat already iterates the elements; avoid constructing
                # an intermediate list.
                flat.extend(nested.data.flat)
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
        # Avoid repeated list concatenations for deep chains like a[1][2][3].
        parts: List[List[Expression]] = []
        current: Expression = expr
        while isinstance(current, IndexExpression):
            parts.append(current.indices)
            current = current.base
        if not parts:
            return current, []
        # Indices are encountered from outermost suffix inward; reverse to
        # preserve evaluation order.
        out: List[Expression] = []
        for chunk in reversed(parts):
            out.extend(chunk)
        return current, out

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
            eval_expr = self._evaluate_expression
            expect_int = self._expect_int
            indices: List[int] = []
            indices_append = indices.append
            for idx_node in index_nodes:
                indices_append(expect_int(eval_expr(idx_node, env), "INDEX", expression.location))
            offset = self._tensor_flat_index(base_val.value, indices, "INDEX", expression.location)
            return base_val.value.data[offset]
        if isinstance(expression, Identifier):
            found = env.get_optional(expression.name)
            if found is not None:
                return found
            if expression.name == "INPUT":
                self._emit_event("before_call", self, "INPUT", [], env, expression.location)
                result = self.builtins.invoke(self, "INPUT", [], [], env, expression.location)
                self._emit_event("after_call", self, "INPUT", result, env, expression.location)
                self._log_step(rule="INPUT", location=expression.location, extra={"args": [], "result": result.value})
                return result
            raise ASMRuntimeError(
                f"Undefined identifier '{expression.name}'",
                location=expression.location,
                rewrite_rule="IDENT",
            )
        if isinstance(expression, CallExpression):
            if expression.name == "IMPORT":
                if any(arg.name for arg in expression.args):
                    raise ASMRuntimeError("IMPORT does not accept keyword arguments", location=expression.location, rewrite_rule="IMPORT")
                first_expr = expression.args[0].expression if expression.args else None
                module_label = first_expr.name if isinstance(first_expr, Identifier) else None
                dummy_args: List[Value] = [Value(TYPE_INT, 0)] * len(expression.args)
                arg_nodes = [arg.expression for arg in expression.args]
                try:
                    self._emit_event("before_call", self, expression.name, [], env, expression.location)
                    result = self.builtins.invoke(self, expression.name, dummy_args, arg_nodes, env, expression.location)
                except ASMRuntimeError:
                    self._log_step(rule="IMPORT", location=expression.location, extra={"module": module_label, "status": "error"})
                    raise
                self._emit_event("after_call", self, expression.name, result, env, expression.location)
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
                    self._emit_event("before_call", self, expression.name, [], env, expression.location)
                    result = self.builtins.invoke(self, expression.name, dummy_args, arg_nodes, env, expression.location)
                except ASMRuntimeError:
                    self._log_step(rule=expression.name, location=expression.location, extra={"args": None, "status": "error"})
                    raise
                self._emit_event("after_call", self, expression.name, result, env, expression.location)
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
            if self.call_stack:
                func_name = self._resolve_user_function_name(frame_name=self.call_stack[-1].name, callee=expression.name)
            else:
                # Shouldn't happen in practice, but preserve behavior.
                func_name = expression.name if expression.name in self.functions else None
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
                self._emit_event("before_call", self, func_name, positional_args, env, expression.location)
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
                self._emit_event("before_call", self, expression.name, positional_args, env, expression.location)
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
            self._emit_event("after_call", self, expression.name, result, env, expression.location)
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
            self._emit_event("after_call", self, function.name, signal.value, env, call_location)
            return signal.value
        except ASMRuntimeError:
            raise
        else:
            self.call_stack.pop()
            if function.return_type == TYPE_INT:
                result = Value(TYPE_INT, 0)
                self._emit_event("after_call", self, function.name, result, env, call_location)
                return result
            if function.return_type == TYPE_FLT:
                result = Value(TYPE_FLT, 0.0)
                self._emit_event("after_call", self, function.name, result, env, call_location)
                return result
            if function.return_type == TYPE_STR:
                result = Value(TYPE_STR, "")
                self._emit_event("after_call", self, function.name, result, env, call_location)
                return result
            if function.return_type == TYPE_TNS:
                raise ASMRuntimeError(
                    f"Function {function.name} must return a tensor value",
                    location=call_location,
                    rewrite_rule=function.name,
                )
            # Extension-defined return types may have defaults; enforce via registry.
            spec = self.type_registry.get_optional(function.return_type)
            if spec is not None and spec.default_value is not None:
                ctx = TypeContext(interpreter=self, location=call_location)
                result = spec.default_value(ctx)
                self._emit_event("after_call", self, function.name, result, env, call_location)
                return result
            raise ASMRuntimeError(
                f"Function {function.name} must return {function.return_type}",
                location=call_location,
                rewrite_rule=function.name,
            )

    def _new_frame(self, name: str, env: Environment, call_location: Optional[SourceLocation]) -> Frame:
        frame_id = f"f_{self.frame_counter:04d}"
        self.frame_counter += 1
        return Frame(name=name, env=env, frame_id=frame_id, call_location=call_location)

    def _emit_event(self, event: str, *args: Any, **kwargs: Any) -> None:
        try:
            self.hook_registry.emit(event, *args, **kwargs)
        except ASMRuntimeError:
            raise
        except Exception as exc:
            loc = None
            if self.logger.entries:
                loc = self.logger.entries[-1].source_location
            raise ASMRuntimeError(
                f"Extension hook '{event}' failed: {exc}",
                location=loc,
                rewrite_rule="EXT",
            )

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
        entry = self.logger.record(
            frame=frame,
            location=location,
            statement=statement,
            env_snapshot=env_snapshot,
            rewrite_record=rewrite,
        )

        # Run extension step rules (every N steps) after recording.
        try:
            self.hook_registry.after_step(
                self,
                StepContext(step_index=entry.step_index, rule=rule, location=location, extra=extra),
            )
        except ASMRuntimeError:
            raise
        except Exception as exc:
            raise ASMRuntimeError(
                f"Extension step rule failed: {exc}",
                location=location,
                rewrite_rule="EXT",
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
