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
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from collections import OrderedDict
from numpy.typing import NDArray

from lexer import ASMError, ASMParseError, Lexer
from extensions import ASMExtensionError, HookRegistry, RuntimeServices, StepContext, TypeContext, TypeRegistry, TypeSpec, build_default_services
from parser import (
    Assignment,
    Declaration,
    Block,
    BreakStatement,
    CallArgument,
    CallExpression,
    Expression,
    ExpressionStatement,
    ForStatement,
    ParForStatement,
    FuncDef,
    LambdaExpression,
    AsyncStatement,
    GotoStatement,
    GotopointStatement,
    Identifier,
    IndexExpression,
    Range,
    Star,
    IfBranch,
    IfStatement,
    Literal,
    MapLiteral,
    Param,
    PointerExpression,
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
    TryStatement,
)


TYPE_INT = "INT"
TYPE_FLT = "FLT"
TYPE_STR = "STR"
TYPE_TNS = "TNS"
TYPE_FUNC = "FUNC"
TYPE_MAP = "MAP"
TYPE_THR = "THR"

# On Windows, command lines over a certain length cause CreateProcess errors
WINDOWS_COMMAND_LENGTH_LIMIT = 8000

@dataclass(frozen=True, slots=True)
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


@dataclass(slots=True)
class Map:
    # Preserve insertion order of keys (left-to-right insertion order).
    data: OrderedDict[Tuple[str, Any], "Value"] = field(default_factory=OrderedDict)


@dataclass(slots=True)
class Value:
    type: str
    value: Any



@dataclass(slots=True)
class PointerRef:
    env: "Environment"
    name: str

    def __repr__(self) -> str:  # pragma: no cover - debug aid only
        return f"<ptr {self.name}>"


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


@dataclass(slots=True)
class Environment:
    parent: Optional["Environment"] = None
    values: Dict[str, Value] = field(default_factory=dict)
    # declared but unassigned symbol types
    declared: Dict[str, str] = field(default_factory=dict)
    frozen: set = field(default_factory=set)
    permafrozen: set = field(default_factory=set)

    def _find_env(self, name: str) -> Optional["Environment"]:
        env: Optional[Environment] = self
        while env is not None:
            if env.values.get(name) is not None:
                return env
            env = env.parent
        return None

    def set(self, name: str, value: Value, declared_type: Optional[str] = None) -> None:
        # Hot-path: inline _find_env to avoid an extra Python call on every assignment.
        env: Optional[Environment] = self
        found_env: Optional[Environment] = None
        found_existing: Optional[Value] = None
        while env is not None:
            values = env.values
            existing = values.get(name)
            if existing is not None:
                found_env = env
                found_existing = existing
                break
            env = env.parent

        if found_env is not None:
            assert found_existing is not None
            existing = found_existing
            if name in found_env.frozen or name in found_env.permafrozen:
                raise ASMRuntimeError(
                    f"Identifier '{name}' is frozen and cannot be reassigned",
                    rewrite_rule="ASSIGN",
                )
            incoming_ptr = value.value if isinstance(value.value, PointerRef) else None
            if incoming_ptr is not None and incoming_ptr.env is found_env and incoming_ptr.name == name:
                raise ASMRuntimeError(
                    "Cannot create self-referential pointer",
                    rewrite_rule="ASSIGN",
                )
            if isinstance(existing.value, PointerRef):
                ptr = existing.value
                if ptr.env is found_env and ptr.name == name:
                    raise ASMRuntimeError(
                        "Cannot assign through self-referential pointer",
                        rewrite_rule="ASSIGN",
                    )
                ptr.env.set(ptr.name, value, declared_type=None)
                return
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
            found_env.values[name] = value
            return

        # No existing value binding found. Look for a prior type declaration
        # in this environment chain.
        decl_env: Optional[Environment] = None
        decl_type: Optional[str] = None
        env2: Optional[Environment] = self
        while env2 is not None:
            if name in env2.declared:
                decl_env = env2
                decl_type = env2.declared[name]
                break
            env2 = env2.parent

        if declared_type is None:
            # Assignment without inline declaration: require a prior declaration
            if decl_env is None:
                raise ASMRuntimeError(
                    f"Identifier '{name}' must be declared with a type before assignment",
                    rewrite_rule="ASSIGN",
                )
            if decl_type != value.type:
                raise ASMRuntimeError(
                    f"Assigned value type {value.type} does not match declaration {decl_type}",
                    rewrite_rule="ASSIGN",
                )
            decl_env.values[name] = value
            return

        # Assignment with inline declaration: if there is an existing declaration
        # ensure it matches; otherwise record declaration in current env.
        if decl_env is not None:
            if decl_type != declared_type:
                raise ASMRuntimeError(
                    f"Type mismatch for '{name}': previously declared as {decl_type}",
                    rewrite_rule="ASSIGN",
                )
            if declared_type != value.type:
                raise ASMRuntimeError(
                    f"Assigned value type {value.type} does not match declaration {declared_type}",
                    rewrite_rule="ASSIGN",
                )
            decl_env.values[name] = value
            return

        # No prior declaration: record it in this environment and create the value
        if declared_type != value.type:
            raise ASMRuntimeError(
                f"Assigned value type {value.type} does not match declaration {declared_type}",
                rewrite_rule="ASSIGN",
            )
        self.declared[name] = declared_type
        self.values[name] = value

    def get(self, name: str) -> Value:
        # Hot-path: inline _find_env to avoid an extra Python call on every read.
        env: Optional[Environment] = self
        while env is not None:
            values = env.values
            found = values.get(name)
            if found is not None:
                return found
            env = env.parent
        raise ASMRuntimeError(f"Undefined identifier '{name}'", rewrite_rule="IDENT")

    def get_optional(self, name: str) -> Optional[Value]:
        # Hot-path: inline _find_env; heavily used by identifier evaluation.
        env: Optional[Environment] = self
        while env is not None:
            values = env.values
            found = values.get(name)
            if found is not None:
                return found
            env = env.parent
        return None

    def delete(self, name: str) -> None:
        env: Optional[Environment] = self
        found_env: Optional[Environment] = None
        while env is not None:
            if env.values.get(name) is not None:
                found_env = env
                break
            env = env.parent

        if found_env is not None:
            if name in found_env.frozen or name in found_env.permafrozen:
                raise ASMRuntimeError(f"Identifier '{name}' is frozen and cannot be deleted", rewrite_rule="DEL")
            del found_env.values[name]
            return
        raise ASMRuntimeError(f"Cannot delete undefined identifier '{name}'", rewrite_rule="DEL")

    def has(self, name: str) -> bool:
        env: Optional[Environment] = self
        while env is not None:
            if env.values.get(name) is not None:
                return True
            env = env.parent
        return False

    def snapshot(self) -> Dict[str, str]:
        def _render(val: Value) -> str:
            if val.type == TYPE_TNS and isinstance(val.value, Tensor):
                dims = ",".join(str(d) for d in val.value.shape)
                return f"{val.type}:[{dims}]"
            if val.type == TYPE_FUNC:
                func_name = getattr(val.value, "name", "<func>")
                return f"{val.type}:{func_name}"
            if isinstance(val.value, PointerRef):
                return f"{val.type}:&{val.value.name}"
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


@dataclass(slots=True)
class Function:
    name: str
    params: List[Param]
    return_type: str
    body: Block
    closure: Environment


@dataclass(slots=True)
class Frame:
    name: str
    env: Environment
    frame_id: str
    call_location: Optional[SourceLocation]
    gotopoints: Dict[Any, int] = field(default_factory=dict)


@dataclass(slots=True)
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


@dataclass(slots=True)
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
        self._register_custom("IMPORT_PATH", 1, 1, self._import_path)
        self._register_custom("RUN", 1, 1, self._run)
        self._register_custom("INPUT", 0, 1, self._input)
        self._register_custom("PRINT", 0, None, self._print)
        self._register_custom("ASSERT", 1, 1, self._assert)
        self._register_custom("THROW", 0, None, self._throw)
        self._register_custom("DEL", 1, 1, self._delete)
        self._register_custom("FREEZE", 1, 1, self._freeze)
        self._register_custom("THAW", 1, 1, self._thaw)
        self._register_custom("PERMAFREEZE", 1, 1, self._permafreeze)
        self._register_custom("FROZEN", 1, 1, self._frozen)
        self._register_custom("PERMAFROZEN", 1, 1, self._permafrozen)
        self._register_custom("EXIST", 1, 1, self._exist)
        self._register_custom("KEYS", 1, 1, self._keys)
        self._register_custom("VALUES", 1, 1, self._values)
        self._register_custom("KEYIN", 2, 2, self._keyin)
        self._register_custom("VALUEIN", 2, 2, self._valuein)
        self._register_custom("EXPORT", 2, 2, self._export)
        self._register_custom("ISINT", 1, 1, self._isint)
        self._register_custom("ISFLT", 1, 1, self._isflt)
        self._register_custom("ISSTR", 1, 1, self._isstr)
        self._register_custom("ISTNS", 1, 1, self._istns)
        self._register_custom("TYPE", 1, 1, self._type)
        self._register_custom("SIGNATURE", 1, 1, self._signature)
        self._register_custom("COPY", 1, 1, self._copy)
        self._register_custom("DEEPCOPY", 1, 1, self._deepcopy)
        self._register_custom("ROUND", 1, 3, self._round)
        self._register_custom("READFILE", 1, 2, self._readfile)
        self._register_custom("BYTES", 1, 2, self._bytes)
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
        self._register_custom("TINT", 1, 1, self._tint)
        self._register_custom("TFLT", 1, 1, self._tflt)
        self._register_custom("TSTR", 1, 1, self._tstr)
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
        self._register_custom("CONV", 2, None, self._convolve)
        self._register_custom("FLIP", 1, 1, self._flip)
        self._register_custom("TFLIP", 2, 2, self._tflip)
        self._register_custom("SCAT", 3, 3, self._scatter)
        self._register_custom("PARALLEL", 1, None, self._parallel)
        self._register_custom("SER", 1, 1, self._serialize)
        self._register_custom("UNSER", 1, 1, self._unserialize)

    def _register_int_only(self, name: str, arity: int, func: Callable[..., int]) -> None:
        if arity == 1:
            def impl(_: "Interpreter", args: List[Value], __: List[Expression], ___: Environment, location: SourceLocation) -> Value:
                a = self._expect_int(args[0], name, location)
                return Value(TYPE_INT, func(a))

            self.table[name] = BuiltinFunction(name=name, min_args=1, max_args=1, impl=impl)
            return
        if arity == 2:
            def impl(_: "Interpreter", args: List[Value], __: List[Expression], ___: Environment, location: SourceLocation) -> Value:
                a = self._expect_int(args[0], name, location)
                b = self._expect_int(args[1], name, location)
                return Value(TYPE_INT, func(a, b))

            self.table[name] = BuiltinFunction(name=name, min_args=2, max_args=2, impl=impl)
            return

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
        value = self._deref_pointer(value, rule=rule, location=location)
        if value.type != TYPE_INT:
            raise ASMRuntimeError(f"{rule} expects integer arguments", location=location, rewrite_rule=rule)
        assert isinstance(value.value, int)
        return value.value

    def _expect_flt(self, value: Value, rule: str, location: SourceLocation) -> float:
        value = self._deref_pointer(value, rule=rule, location=location)
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
        return Value(TYPE_FLT, float(math.lcm(int(a), int(b))))

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
        value = self._deref_pointer(value, rule=rule, location=location)
        if value.type != TYPE_STR:
            raise ASMRuntimeError(f"{rule} expects string arguments", location=location, rewrite_rule=rule)
        assert isinstance(value.value, str)
        return value.value

    def _expect_tns(self, value: Value, rule: str, location: SourceLocation) -> Tensor:
        value = self._deref_pointer(value, rule=rule, location=location)
        if value.type != TYPE_TNS:
            raise ASMRuntimeError(f"{rule} expects tensor arguments", location=location, rewrite_rule=rule)
        assert isinstance(value.value, Tensor)
        return value.value

    def _deref_pointer(self, value: Value, *, rule: str, location: SourceLocation) -> Value:
        current = value
        hops = 0
        while isinstance(current.value, PointerRef):
            hops += 1
            if hops > 128:
                raise ASMRuntimeError("Pointer cycle detected", location=location, rewrite_rule=rule)
            ptr = current.value
            target = ptr.env.get_optional(ptr.name)
            if target is None:
                raise ASMRuntimeError(
                    f"Pointer target '{ptr.name}' is undefined",
                    location=location,
                    rewrite_rule=rule,
                )
            current = target
        return current

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
            return Value(TYPE_INT, math.prod(ints))
        if first_type == TYPE_FLT:
            flts = [self._expect_flt(v, "PROD", location) for v in values]
            return Value(TYPE_FLT, float(math.prod(flts)))
        raise ASMRuntimeError("PROD expects INT or FLT arguments", location=location, rewrite_rule="PROD")

    def _max(self, _: "Interpreter", values: List[Value], location: SourceLocation) -> Value:
        if not values:
            raise ASMRuntimeError("MAX requires at least one argument", rewrite_rule="MAX")
        first_type = values[0].type
        # Allow MAX over tensors: flatten all tensor arguments and compute
        # the maximum element. All elements must have the same (scalar)
        # type. Tensor-of-tensors or mixed element types are rejected.
        if first_type == TYPE_TNS:
            # Ensure all args are tensors
            if any(v.type != TYPE_TNS for v in values):
                raise ASMRuntimeError("MAX cannot mix values of different types", rewrite_rule="MAX", location=location)
            # Gather all elements from all tensors (flatten)
            elems: List[Value] = []
            for v in values:
                tensor = self._expect_tns(v, "MAX", location)
                # tensor.data is a numpy array of Value objects
                for el in tensor.data.ravel():
                    # Dereference any pointer values inside the tensor
                    elems.append(self._deref_pointer(el, rule="MAX", location=location))
            if not elems:
                raise ASMRuntimeError("MAX requires tensors with at least one element", rewrite_rule="MAX", location=location)
            # All elements must share the same type
            elem_type = elems[0].type
            # Elements must be scalar values; tensors-as-elements are forbidden
            if elem_type == TYPE_TNS or any(e.type == TYPE_TNS for e in elems):
                raise ASMRuntimeError("MAX tensor elements must be scalar (INT, FLT, or STR)", rewrite_rule="MAX", location=location)
            if any(e.type != elem_type for e in elems):
                raise ASMRuntimeError("MAX cannot mix values of different types", rewrite_rule="MAX", location=location)
            if elem_type == TYPE_INT:
                ints = [self._expect_int(e, "MAX", location) for e in elems]
                return Value(TYPE_INT, max(ints))
            if elem_type == TYPE_FLT:
                flts = [self._expect_flt(e, "MAX", location) for e in elems]
                return Value(TYPE_FLT, float(max(flts)))
            if elem_type == TYPE_STR:
                strs = [self._expect_str(e, "MAX", location) for e in elems]
                longest = max(strs, key=len)
                return Value(TYPE_STR, longest)
            raise ASMRuntimeError("MAX cannot operate on tensors of this element type", rewrite_rule="MAX", location=location)

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
        # Allow MIN over tensors: flatten all tensor arguments and compute
        # the minimum element. All elements must have the same (scalar)
        # type. Tensor-of-tensors or mixed element types are rejected.
        if first_type == TYPE_TNS:
            # Ensure all args are tensors
            if any(v.type != TYPE_TNS for v in values):
                raise ASMRuntimeError("MIN cannot mix values of different types", rewrite_rule="MIN", location=location)
            # Gather all elements from all tensors (flatten)
            elems: List[Value] = []
            for v in values:
                tensor = self._expect_tns(v, "MIN", location)
                for el in tensor.data.ravel():
                    elems.append(self._deref_pointer(el, rule="MIN", location=location))
            if not elems:
                raise ASMRuntimeError("MIN requires tensors with at least one element", rewrite_rule="MIN", location=location)
            # Elements must be scalar values; tensors-as-elements are forbidden
            elem_type = elems[0].type
            if elem_type == TYPE_TNS or any(e.type == TYPE_TNS for e in elems):
                raise ASMRuntimeError("MIN tensor elements must be scalar (INT, FLT, or STR)", rewrite_rule="MIN", location=location)
            if any(e.type != elem_type for e in elems):
                raise ASMRuntimeError("MIN cannot mix values of different types", rewrite_rule="MIN", location=location)
            if elem_type == TYPE_INT:
                ints = [self._expect_int(e, "MIN", location) for e in elems]
                return Value(TYPE_INT, min(ints))
            if elem_type == TYPE_FLT:
                flts = [self._expect_flt(e, "MIN", location) for e in elems]
                return Value(TYPE_FLT, float(min(flts)))
            if elem_type == TYPE_STR:
                strs = [self._expect_str(e, "MIN", location) for e in elems]
                shortest = min(strs, key=len)
                return Value(TYPE_STR, shortest)
            raise ASMRuntimeError("MIN cannot operate on tensors of this element type", rewrite_rule="MIN", location=location)

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

    def _keys(
        self,
        _: "Interpreter",
        args: List[Value],
        __: List[Expression],
        ___: Environment,
        location: SourceLocation,
    ) -> Value:
        # KEYS(MAP: map):TNS -> returns 1-D tensor of the map's keys in insertion order
        val = args[0]
        if val.type != TYPE_MAP:
            raise ASMRuntimeError("KEYS expects a MAP argument", location=location, rewrite_rule="KEYS")
        m = val.value
        assert isinstance(m, Map)
        out: List[Value] = []
        for key_type, key_val in m.data.keys():
            if key_type == TYPE_INT:
                out.append(Value(TYPE_INT, key_val))
            elif key_type == TYPE_FLT:
                out.append(Value(TYPE_FLT, key_val))
            elif key_type == TYPE_STR:
                out.append(Value(TYPE_STR, key_val))
            else:
                raise ASMRuntimeError("Unsupported map key type", location=location, rewrite_rule="KEYS")
        arr = np.array(out, dtype=object)
        return Value(TYPE_TNS, Tensor(shape=[len(out)], data=arr))

    def _values(
        self,
        _: "Interpreter",
        args: List[Value],
        __: List[Expression],
        ___: Environment,
        location: SourceLocation,
    ) -> Value:
        # VALUES(MAP: map):TNS -> returns 1-D tensor of the map's values in insertion order
        val = args[0]
        if val.type != TYPE_MAP:
            raise ASMRuntimeError("VALUES expects a MAP argument", location=location, rewrite_rule="VALUES")
        m = val.value
        assert isinstance(m, Map)
        out: List[Value] = [v for v in m.data.values()]
        arr = np.array(out, dtype=object)
        return Value(TYPE_TNS, Tensor(shape=[len(out)], data=arr))

    def _keyin(
        self,
        _: "Interpreter",
        args: List[Value],
        __: List[Expression],
        ___: Environment,
        location: SourceLocation,
    ) -> Value:
        # KEYIN(INT|FLT|STR: key, MAP: map):INT -> 1 if the key exists in map
        if len(args) != 2:
            raise ASMRuntimeError("KEYIN requires two arguments", location=location, rewrite_rule="KEYIN")
        key = args[0]
        mval = args[1]
        if mval.type != TYPE_MAP:
            raise ASMRuntimeError("KEYIN expects a MAP as second argument", location=location, rewrite_rule="KEYIN")
        if key.type not in (TYPE_INT, TYPE_FLT, TYPE_STR):
            raise ASMRuntimeError("KEYIN expects key of type INT, FLT, or STR", location=location, rewrite_rule="KEYIN")
        m = mval.value
        assert isinstance(m, Map)
        exists = (key.type, key.value) in m.data
        return Value(TYPE_INT, 1 if exists else 0)

    def _valuein(
        self,
        interpreter: "Interpreter",
        args: List[Value],
        __: List[Expression],
        ___: Environment,
        location: SourceLocation,
    ) -> Value:
        # VALUEIN(ANY: value, MAP: map):INT -> 1 if any map value equals value
        if len(args) != 2:
            raise ASMRuntimeError("VALUEIN requires two arguments", location=location, rewrite_rule="VALUEIN")
        needle = args[0]
        mval = args[1]
        if mval.type != TYPE_MAP:
            raise ASMRuntimeError("VALUEIN expects a MAP as second argument", location=location, rewrite_rule="VALUEIN")
        m = mval.value
        assert isinstance(m, Map)
        for v in m.data.values():
            if interpreter._values_equal(needle, v):
                return Value(TYPE_INT, 1)
        return Value(TYPE_INT, 0)

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
        if root == "<repl>":
            return Value(TYPE_INT, 1 if location.file == "<repl>" else 0)
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

        base_dir = os.getcwd() if location.file == "<repl>" else os.path.dirname(os.path.abspath(location.file))
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
        source_lines = source_text.splitlines()
        parser = Parser(tokens, module_path, source_lines, type_names=interpreter.type_registry.names())
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

    def _import_path(
        self,
        interpreter: "Interpreter",
        args: List[Value],
        arg_nodes: List[Expression],
        env: Environment,
        location: SourceLocation,
    ) -> Value:
        # IMPORT_PATH(path): import the module located at absolute filesystem path `path`.
        path = self._expect_str(args[0], "IMPORT_PATH", location)
        if not os.path.isabs(path):
            raise ASMRuntimeError("IMPORT_PATH expects an absolute path", location=location, rewrite_rule="IMPORT_PATH")

        module_path = os.path.abspath(path)
        if not os.path.exists(module_path):
            raise ASMRuntimeError(f"Module file not found: {module_path}", location=location, rewrite_rule="IMPORT_PATH")

        module_name = os.path.splitext(os.path.basename(module_path))[0]

        pre_function_keys = set(interpreter.functions.keys())

        # If module was already imported by name, reuse cached env/functions.
        if module_name in interpreter.module_cache:
            cached_env = interpreter.module_cache[module_name]
            for fn in interpreter.module_functions.get(module_name, []):
                if fn.name not in interpreter.functions:
                    interpreter.functions[fn.name] = fn

            for k, v in cached_env.values.items():
                dotted = f"{module_name}.{k}"
                env.set(dotted, v, declared_type=v.type)

            if set(interpreter.functions.keys()) != pre_function_keys:
                interpreter._mark_functions_changed()
            return Value(TYPE_INT, 0)

        try:
            with open(module_path, "r", encoding="utf-8") as handle:
                source_text = handle.read()
        except OSError as exc:
            raise ASMRuntimeError(f"Failed to import path '{module_path}': {exc}", location=location, rewrite_rule="IMPORT_PATH")

        # --- extension loading (same behavior as IMPORT) ---
        try:
            import extensions as _extmod

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
                    raise ASMExtensionError(f"Failed to load extensions from {companion_asmxt}: {exc}") from exc

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
            raise ASMRuntimeError(str(exc), location=location, rewrite_rule="IMPORT_PATH")
        except Exception:
            pass

        lexer = Lexer(source_text, module_path)
        tokens = lexer.tokenize()
        source_lines = source_text.splitlines()
        parser = Parser(tokens, module_path, source_lines, type_names=interpreter.type_registry.names())
        program = parser.parse()

        module_env = Environment()
        prev_functions = dict(interpreter.functions)
        try:
            interpreter._execute_block(program.statements, module_env)
        except Exception as exc:
            interpreter.functions = prev_functions
            if isinstance(exc, ASMRuntimeError):
                raise
            raise ASMRuntimeError(f"Import failed: {exc}", location=location, rewrite_rule="IMPORT_PATH")

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

        for fn in registered_functions:
            parts = fn.name.split(".", 1)
            if len(parts) == 2:
                _, unqualified = parts
                alias_name = f"{module_name}.{unqualified}"
                if alias_name not in interpreter.functions:
                    alias_fn = Function(
                        name=alias_name,
                        params=fn.params,
                        return_type=fn.return_type,
                        body=fn.body,
                        closure=module_env,
                    )
                    interpreter.functions[alias_name] = alias_fn

        interpreter.module_cache[module_name] = module_env
        interpreter.module_functions[module_name] = registered_functions

        if set(interpreter.functions.keys()) != pre_function_keys:
            interpreter._mark_functions_changed()

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
        source_lines = source_text.splitlines()
        parser = Parser(tokens, run_filename, source_lines, type_names=interpreter.type_registry.names())
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

    def _throw(
        self,
        interpreter: "Interpreter",
        args: List[Value],
        __: List[Expression],
        ___: Environment,
        location: SourceLocation,
    ) -> Value:
        # Render arguments the same way PRINT does, then raise with the
        # concatenated string as the error message.
        rendered: List[str] = []
        for arg in args:
            if arg.type == TYPE_INT:
                number = self._expect_int(arg, "THROW", location)
                rendered.append(("-" + format(-number, "b")) if number < 0 else format(number, "b"))
            elif arg.type == TYPE_STR:
                rendered.append(arg.value)  # type: ignore[arg-type]
            else:
                spec = interpreter.type_registry.get_optional(arg.type)
                if spec is not None and spec.printable:
                    ctx = TypeContext(interpreter=interpreter, location=location)
                    rendered.append(spec.to_str(ctx, arg))
                else:
                    raise ASMRuntimeError(
                        "THROW accepts INT or STR arguments",
                        location=location,
                        rewrite_rule="THROW",
                    )
        msg = "".join(rendered)
        raise ASMRuntimeError(msg, location=location, rewrite_rule="THROW")

    def _delete(
        self,
        interpreter: "Interpreter",
        args: List[Value],
        arg_nodes: List[Expression],
        env: Environment,
        location: SourceLocation,
    ) -> Value:
        if not arg_nodes:
            raise ASMRuntimeError("DEL expects identifier or indexed target", location=location, rewrite_rule="DEL")
        first = arg_nodes[0]
        # Deleting a top-level identifier
        if isinstance(first, Identifier):
            name = first.name
            try:
                env.delete(name)
            except ASMRuntimeError as err:
                err.location = location
                raise
            return Value(TYPE_INT, 0)

        # Deleting a map key via indexed expression, e.g. DEL(map<k>)
        if isinstance(first, IndexExpression):
            base_expr, index_nodes = interpreter._gather_index_chain(first)
            if not isinstance(base_expr, Identifier):
                raise ASMRuntimeError("DEL expects identifier base for indexed deletion", location=location, rewrite_rule="DEL")
            base_val = interpreter._evaluate_expression(base_expr, env)
            if base_val.type != TYPE_MAP:
                raise ASMRuntimeError("DEL indexed deletion requires a map base", location=location, rewrite_rule="DEL")
            assert isinstance(base_val.value, Map)
            eval_expr = interpreter._evaluate_expression
            current_map = base_val.value
            # Traverse to parent of final key
            for node in index_nodes[:-1]:
                key_val = eval_expr(node, env)
                if key_val.type not in (TYPE_INT, TYPE_FLT, TYPE_STR):
                    raise ASMRuntimeError("Map keys must be INT, FLT, or STR", location=location, rewrite_rule="DEL")
                key = (key_val.type, key_val.value)
                if key not in current_map.data:
                    raise ASMRuntimeError(f"Key not found: {key_val.value}", location=location, rewrite_rule="DEL")
                next_val = current_map.data[key]
                if next_val.type != TYPE_MAP:
                    raise ASMRuntimeError("Cannot traverse non-map value during DEL", location=location, rewrite_rule="DEL")
                current_map = next_val.value
            # Final key
            final_node = index_nodes[-1]
            final_key_val = eval_expr(final_node, env)
            if final_key_val.type not in (TYPE_INT, TYPE_FLT, TYPE_STR):
                raise ASMRuntimeError("Map keys must be INT, FLT, or STR", location=location, rewrite_rule="DEL")
            final_key = (final_key_val.type, final_key_val.value)
            if final_key not in current_map.data:
                raise ASMRuntimeError(f"Key not found: {final_key_val.value}", location=location, rewrite_rule="DEL")
            del current_map.data[final_key]
            return Value(TYPE_INT, 0)

        raise ASMRuntimeError("DEL expects identifier or indexed target", location=location, rewrite_rule="DEL")

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

    def _signature(
        self,
        interpreter: "Interpreter",
        args: List[Value],
        arg_nodes: List[Expression],
        env: Environment,
        location: SourceLocation,
    ) -> Value:
        # SIGNATURE(SYMBOL: sym):STR -> return a textual signature for sym
        if len(args) != 1:
            raise ASMRuntimeError("SIGNATURE expects one argument", location=location, rewrite_rule="SIGNATURE")
        if not arg_nodes or not isinstance(arg_nodes[0], Identifier):
            raise ASMRuntimeError("SIGNATURE requires an identifier argument", location=location, rewrite_rule="SIGNATURE")
        name = arg_nodes[0].name

        # First prefer an environment binding
        try:
            if env.has(name):
                val = env.get(name)
                if val.type == TYPE_FUNC:
                    func = val.value
                else:
                    return Value(TYPE_STR, f"{val.type}: {name}")
            elif name in interpreter.functions:
                func = interpreter.functions[name]
            else:
                raise ASMRuntimeError(f"Unknown identifier '{name}'", location=location, rewrite_rule="SIGNATURE")
        except ASMRuntimeError:
            raise
        except Exception as exc:
            raise ASMRuntimeError(str(exc), location=location, rewrite_rule="SIGNATURE")

        # Build function signature in the documented form:
        # FUNC name(T1:arg1, T2:arg2 = def, ...):R
        parts: List[str] = []
        for p in func.params:
            part = f"{p.type}: {p.name}"
            if p.default is not None:
                # Try to render simple literal defaults
                d = p.default
                rendered = "<expr>"
                try:
                    if isinstance(d, Literal):
                        if d.literal_type == "INT":
                            ival = int(d.value)
                            if ival < 0:
                                rendered = "-" + bin(abs(ival))[2:]
                            else:
                                rendered = bin(ival)[2:]
                        elif d.literal_type == "STR":
                            rendered = '"' + str(d.value) + '"'
                        else:
                            rendered = repr(d.value)
                    else:
                        # best-effort: use source snippet if available
                        loc = getattr(d, "location", None)
                        if loc is not None and getattr(loc, "statement", ""):
                            stmt = loc.statement
                            col = max(1, getattr(loc, "column", 1))
                            # columns are 1-based; slice defensively
                            try:
                                rendered = stmt[col - 1 :].strip()
                            except Exception:
                                rendered = stmt
                except Exception:
                    rendered = "<expr>"
                part += " = " + rendered
            parts.append(part)

        params_text = ", ".join(parts)
        sig = f"FUNC {func.name}({params_text}):{func.return_type}"
        return Value(TYPE_STR, sig)

    def _copy(
        self,
        interpreter: "Interpreter",
        args: List[Value],
        __: List[Expression],
        ___: Environment,
        location: SourceLocation,
    ) -> Value:
        # COPY(ANY: obj):ANY -> shallow copy of obj. For primitives this
        # returns a same-typed Value wrapper; for TNS/MAP it returns a new
        # container with element references preserved.
        if len(args) != 1:
            raise ASMRuntimeError("COPY expects one argument", location=location, rewrite_rule="COPY")
        val = args[0]
        if val.type in (TYPE_INT, TYPE_FLT, TYPE_STR, TYPE_FUNC):
            return Value(val.type, val.value)
        if val.type == TYPE_TNS:
            tensor = self._expect_tns(val, "COPY", location)
            flat = tensor.data.ravel()
            n = flat.size
            out = np.empty(n, dtype=object)
            for i in range(n):
                out[i] = flat[i]
            return Value(TYPE_TNS, Tensor(shape=list(tensor.shape), data=out))
        if val.type == TYPE_MAP:
            m = val.value
            assert isinstance(m, Map)
            new_map = Map()
            for k, v in m.data.items():
                new_map.data[k] = v
            return Value(TYPE_MAP, new_map)
        raise ASMRuntimeError("COPY: unsupported value type", location=location, rewrite_rule="COPY")

    def _deep_copy_value(self, interpreter: "Interpreter", val: Value, location: SourceLocation) -> Value:
        # Helper for recursive deep-copy of Value objects
        if val.type in (TYPE_INT, TYPE_FLT, TYPE_STR, TYPE_FUNC):
            return Value(val.type, val.value)
        if val.type == TYPE_TNS:
            tensor = self._expect_tns(val, "DEEPCOPY", location)
            flat = tensor.data.ravel()
            new_list: List[Value] = [self._deep_copy_value(interpreter, item, location) for item in flat]
            arr = np.array(new_list, dtype=object)
            return Value(TYPE_TNS, Tensor(shape=list(tensor.shape), data=arr))
        if val.type == TYPE_MAP:
            m = val.value
            assert isinstance(m, Map)
            new_map = Map()
            for k, v in m.data.items():
                new_map.data[k] = self._deep_copy_value(interpreter, v, location)
            return Value(TYPE_MAP, new_map)
        raise ASMRuntimeError("DEEPCOPY: unsupported value type", location=location, rewrite_rule="DEEPCOPY")

    def _deepcopy(
        self,
        interpreter: "Interpreter",
        args: List[Value],
        __: List[Expression],
        ___: Environment,
        location: SourceLocation,
    ) -> Value:
        # DEEPCOPY(ANY: obj):ANY -> recursively copy containers so elements
        # are freshly-copied as well.
        if len(args) != 1:
            raise ASMRuntimeError("DEEPCOPY expects one argument", location=location, rewrite_rule="DEEPCOPY")
        return self._deep_copy_value(interpreter, args[0], location)

    def _serialize(
        self,
        interpreter: "Interpreter",
        args: List[Value],
        __: List[Expression],
        ___: Environment,
        location: SourceLocation,
    ) -> Value:
        # SER(ANY: obj):STR -> JSON text representing typed value
        if len(args) != 1:
            raise ASMRuntimeError("SER expects one argument", location=location, rewrite_rule="SER")

        def ser_val(v: Value):
            t = v.type
            if t == TYPE_INT:
                assert isinstance(v.value, int)
                n = v.value
                if n == 0:
                    digits = "0"
                else:
                    digits = bin(abs(n))[2:]
                return {"t": "INT", "v": ("-" + digits) if n < 0 else digits}
            if t == TYPE_FLT:
                assert isinstance(v.value, float)
                return {"t": "FLT", "v": repr(v.value)}
            if t == TYPE_STR:
                assert isinstance(v.value, str)
                return {"t": "STR", "v": v.value}
            if t == TYPE_TNS:
                tensor = self._expect_tns(v, "SER", location)
                flat = tensor.data.ravel()
                serialized = [ser_val(item) for item in flat]
                return {"t": "TNS", "shape": list(tensor.shape), "v": serialized}
            if t == TYPE_MAP:
                m = v.value
                assert isinstance(m, Map)
                items = []
                for k, vv in m.data.items():
                    # k is (type, value)
                    kt, kv = k
                    if kt == TYPE_INT:
                        if kv == 0:
                            krep = "0"
                        else:
                            krep = ("-" + bin(abs(kv))[2:]) if kv < 0 else bin(kv)[2:]
                        key_ser = {"t": "INT", "v": krep}
                    elif kt == TYPE_FLT:
                        key_ser = {"t": "FLT", "v": repr(kv)}
                    elif kt == TYPE_STR:
                        key_ser = {"t": "STR", "v": kv}
                    else:
                        raise ASMRuntimeError("SER: unsupported map key type", location=location, rewrite_rule="SER")
                    items.append({"k": key_ser, "v": ser_val(vv)})
                return {"t": "MAP", "v": items}
            # For FUNC and THR and other values, emit a descriptive form.
            return {"t": t, "repr": str(v.value)}

        try:
            body = ser_val(args[0])
            text = json.dumps(body, separators=(",", ":"))
            return Value(TYPE_STR, text)
        except ASMRuntimeError:
            raise
        except Exception as exc:
            raise ASMRuntimeError(f"SER failed: {exc}", location=location, rewrite_rule="SER")

    def _unserialize(
        self,
        interpreter: "Interpreter",
        args: List[Value],
        __: List[Expression],
        ___: Environment,
        location: SourceLocation,
    ) -> Value:
        # UNSER(STR: obj):ANY -> reverse of SER
        if len(args) != 1:
            raise ASMRuntimeError("UNSER expects one argument", location=location, rewrite_rule="UNSER")
        s = args[0]
        text = self._expect_str(s, "UNSER", location)

        def deser_val(obj) -> Value:
            if not isinstance(obj, dict) or "t" not in obj:
                raise ASMRuntimeError("UNSER: invalid serialized form", location=location, rewrite_rule="UNSER")
            t = obj["t"]
            if t == "INT":
                raw = obj.get("v", "0")
                neg = False
                if isinstance(raw, str) and raw.startswith("-"):
                    neg = True
                    raw = raw[1:]
                if raw == "":
                    ival = 0
                else:
                    ival = int(raw, 2)
                if neg:
                    ival = -ival
                return Value(TYPE_INT, ival)
            if t == "FLT":
                raw = obj.get("v", "0.0")
                try:
                    f = float(raw)
                except Exception:
                    raise ASMRuntimeError("UNSER: invalid FLT literal", location=location, rewrite_rule="UNSER")
                return Value(TYPE_FLT, f)
            if t == "STR":
                return Value(TYPE_STR, obj.get("v", ""))
            if t == "TNS":
                shape = obj.get("shape")
                flat = obj.get("v", [])
                if not isinstance(shape, list):
                    raise ASMRuntimeError("UNSER: invalid TNS shape", location=location, rewrite_rule="UNSER")
                vals = [deser_val(x) for x in flat]
                arr = np.array(vals, dtype=object)
                return Value(TYPE_TNS, Tensor(shape=list(shape), data=arr))
            if t == "MAP":
                items = obj.get("v", [])
                if not isinstance(items, list):
                    raise ASMRuntimeError("UNSER: invalid MAP body", location=location, rewrite_rule="UNSER")
                m = Map()
                for pair in items:
                    key_obj = pair.get("k")
                    val_obj = pair.get("v")
                    if not isinstance(key_obj, dict) or "t" not in key_obj:
                        raise ASMRuntimeError("UNSER: invalid MAP key", location=location, rewrite_rule="UNSER")
                    kt = key_obj["t"]
                    if kt == "INT":
                        raw = key_obj.get("v", "0")
                        neg = False
                        if isinstance(raw, str) and raw.startswith("-"):
                            neg = True
                            raw = raw[1:]
                        kval = int(raw, 2) if raw != "" else 0
                        if neg:
                            kval = -kval
                    elif kt == "FLT":
                        kval = float(key_obj.get("v", 0.0))
                    elif kt == "STR":
                        kval = key_obj.get("v", "")
                    else:
                        raise ASMRuntimeError("UNSER: unsupported MAP key type", location=location, rewrite_rule="UNSER")
                    vval = deser_val(val_obj)
                    m.data[(kt, kval)] = vval
                return Value(TYPE_MAP, m)
            # FUNC / THR cannot be reconstructed reliably
            raise ASMRuntimeError(f"UNSER: cannot reconstruct type {t}", location=location, rewrite_rule="UNSER")

        try:
            obj = json.loads(text)
        except Exception:
            raise ASMRuntimeError("UNSER: invalid JSON", location=location, rewrite_rule="UNSER")
        return deser_val(obj)

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
        # Optional endian argument: default is big-endian
        endian = "big"
        if len(args) >= 2:
            endian_val = args[1]
            if endian_val.type != TYPE_STR:
                raise ASMRuntimeError("BYTES expects string endian argument", location=location, rewrite_rule="BYTES")
            endian = str(endian_val.value).strip().lower()
        # Validate endian
        if endian not in {"big", "little"}:
            raise ASMRuntimeError("BYTES endian must be 'big' or 'little'", location=location, rewrite_rule="BYTES")

        if number == 0:
            data = np.array([Value(TYPE_INT, 0)], dtype=object)
            return Value(TYPE_TNS, Tensor(shape=[1], data=data))

        octets: List[int] = []
        temp = number
        while temp > 0:
            octets.append(temp & 0xFF)
            temp >>= 8
        # octets currently little-endian (LSB first); reverse for big-endian
        if endian == "big":
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
        # Use ravel() and indexed loop to avoid per-iteration tuple allocation
        x_flat = x.data.ravel()
        y_flat = y.data.ravel()
        n = x_flat.size
        out = np.empty(n, dtype=object)
        # Validate element types once and then access .value directly
        if n == 0:
            return Tensor(shape=list(x.shape), data=np.array([], dtype=object))
        if x_flat[0].type != TYPE_INT:
            raise ASMRuntimeError(f"{rule} expects INT tensor elements", location=location, rewrite_rule=rule)
        for v in x_flat:
            if v.type != TYPE_INT:
                raise ASMRuntimeError(f"{rule} tensor has mixed element types", location=location, rewrite_rule=rule)
        for v in y_flat:
            if v.type != TYPE_INT:
                raise ASMRuntimeError(f"{rule} tensor has mixed element types", location=location, rewrite_rule=rule)
        for i in range(n):
            a = int(x_flat[i].value)
            b = int(y_flat[i].value)
            out[i] = Value(TYPE_INT, op(a, b))
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
        # Use flattened views and indexed loops for better performance
        x_flat = x.data.ravel()
        y_flat = y.data.ravel()
        first_type = x_flat[0].type
        if first_type not in (TYPE_INT, TYPE_FLT):
            raise ASMRuntimeError(f"{rule} expects INT or FLT tensor elements", location=location, rewrite_rule=rule)
        # Validate uniform element types in both tensors
        for v in x_flat:
            if v.type != first_type:
                raise ASMRuntimeError(f"{rule} cannot mix INT and FLT", location=location, rewrite_rule=rule)
        for v in y_flat:
            if v.type != first_type:
                raise ASMRuntimeError(f"{rule} cannot mix INT and FLT", location=location, rewrite_rule=rule)

        n = x_flat.size
        out = np.empty(n, dtype=object)
        if first_type == TYPE_INT:
            # Validate and then access int values directly to avoid repeated checks
            for v in x_flat:
                if v.type != TYPE_INT:
                    raise ASMRuntimeError(f"{rule} cannot mix INT and FLT", location=location, rewrite_rule=rule)
            for v in y_flat:
                if v.type != TYPE_INT:
                    raise ASMRuntimeError(f"{rule} cannot mix INT and FLT", location=location, rewrite_rule=rule)
            for i in range(n):
                a = int(x_flat[i].value)
                b = int(y_flat[i].value)
                out[i] = Value(TYPE_INT, op_int(a, b))
            return Tensor(shape=list(x.shape), data=out)

        for i in range(n):
            a = x_flat[i]
            b = y_flat[i]
            out[i] = Value(TYPE_FLT, op_flt(float(a.value), float(b.value)))
        return Tensor(shape=list(x.shape), data=out)

    def _map_tensor_int_scalar(self, tensor: Tensor, scalar: int, rule: str, location: SourceLocation, op: Callable[[int, int], int]) -> Tensor:
        flat = tensor.data.ravel()
        n = flat.size
        out = np.empty(n, dtype=object)
        # Validate element types once, then use direct int access
        if n == 0:
            return Tensor(shape=list(tensor.shape), data=np.array([], dtype=object))
        if flat[0].type != TYPE_INT:
            raise ASMRuntimeError(f"{rule} expects INT tensor elements", location=location, rewrite_rule=rule)
        for v in flat:
            if v.type != TYPE_INT:
                raise ASMRuntimeError(f"{rule} tensor has mixed element types", location=location, rewrite_rule=rule)
        for i in range(n):
            entry = int(flat[i].value)
            out[i] = Value(TYPE_INT, op(entry, scalar))
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
        flat = tensor.data.ravel()
        first_type = flat[0].type
        if first_type not in (TYPE_INT, TYPE_FLT):
            raise ASMRuntimeError(f"{rule} expects INT or FLT tensor elements", location=location, rewrite_rule=rule)
        if scalar.type != first_type:
            raise ASMRuntimeError(f"{rule} cannot mix INT and FLT", location=location, rewrite_rule=rule)
        for v in flat:
            if v.type != first_type:
                raise ASMRuntimeError(f"{rule} tensor has mixed element types", location=location, rewrite_rule=rule)

        n = flat.size
        out = np.empty(n, dtype=object)
        if first_type == TYPE_INT:
            sc = int(scalar.value)
            for i in range(n):
                entry = int(flat[i].value)
                out[i] = Value(TYPE_INT, op_int(entry, sc))
            return Tensor(shape=list(tensor.shape), data=out)
        scf = float(scalar.value)
        for i in range(n):
            entry = float(flat[i].value)
            out[i] = Value(TYPE_FLT, op_flt(entry, scf))
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

    def _tint(
        self,
        interpreter: "Interpreter",
        args: List[Value],
        __: List[Expression],
        ___: Environment,
        location: SourceLocation,
    ) -> Value:
        """TINT(TNS: obj):TNS
        Convert each element of `obj` to INT using `INT` conversion rules.
        Raises a runtime error if any element cannot be converted.
        """
        tensor = self._expect_tns(args[0], "TINT", location)
        flat = tensor.data.ravel()
        n = flat.size
        if n == 0:
            return Value(TYPE_TNS, Tensor(shape=list(tensor.shape), data=np.array([], dtype=object)))
        out = np.empty(n, dtype=object)
        for i in range(n):
            entry = flat[i]
            try:
                converted = self._int_op(None, [entry], [], None, location)
            except ASMRuntimeError as e:
                raise ASMRuntimeError(f"TINT: cannot convert tensor element: {e.message}", location=location, rewrite_rule="TINT")
            out[i] = converted
        return Value(TYPE_TNS, Tensor(shape=list(tensor.shape), data=out))

    def _tflt(
        self,
        interpreter: "Interpreter",
        args: List[Value],
        __: List[Expression],
        ___: Environment,
        location: SourceLocation,
    ) -> Value:
        """TFLT(TNS: obj):TNS
        Convert each element of `obj` to FLT using `FLT` conversion rules.
        Raises a runtime error if any element cannot be converted.
        """
        tensor = self._expect_tns(args[0], "TFLT", location)
        flat = tensor.data.ravel()
        n = flat.size
        if n == 0:
            return Value(TYPE_TNS, Tensor(shape=list(tensor.shape), data=np.array([], dtype=object)))
        out = np.empty(n, dtype=object)
        for i in range(n):
            entry = flat[i]
            try:
                converted = self._flt_op(None, [entry], [], None, location)
            except ASMRuntimeError as e:
                raise ASMRuntimeError(f"TFLT: cannot convert tensor element: {e.message}", location=location, rewrite_rule="TFLT")
            out[i] = converted
        return Value(TYPE_TNS, Tensor(shape=list(tensor.shape), data=out))

    def _tstr(
        self,
        interpreter: "Interpreter",
        args: List[Value],
        __: List[Expression],
        ___: Environment,
        location: SourceLocation,
    ) -> Value:
        """TSTR(TNS: obj):TNS
        Convert each element of `obj` to STR using `STR` conversion rules.
        Raises a runtime error if any element cannot be converted.
        """
        tensor = self._expect_tns(args[0], "TSTR", location)
        flat = tensor.data.ravel()
        n = flat.size
        if n == 0:
            return Value(TYPE_TNS, Tensor(shape=list(tensor.shape), data=np.array([], dtype=object)))
        out = np.empty(n, dtype=object)
        for i in range(n):
            entry = flat[i]
            try:
                converted = self._str_op(None, [entry], [], None, location)
            except ASMRuntimeError as e:
                raise ASMRuntimeError(f"TSTR: cannot convert tensor element: {e.message}", location=location, rewrite_rule="TSTR")
            out[i] = converted
        return Value(TYPE_TNS, Tensor(shape=list(tensor.shape), data=out))

    def _scatter(
        self,
        interpreter: "Interpreter",
        args: List[Value],
        __: List[Expression],
        ___: Environment,
        location: SourceLocation,
    ) -> Value:
        """SCAT(TNS: src, TNS: dst, TNS: ind):TNS

        Copy `src` into a slice of `dst` defined by per-dimension [lo, hi] pairs.
        Returns a new tensor shaped like `dst` with the slice replaced.
        """

        src = self._expect_tns(args[0], "SCAT", location)
        dst = self._expect_tns(args[1], "SCAT", location)
        ind = self._expect_tns(args[2], "SCAT", location)

        rank = len(dst.shape)
        if len(src.shape) != rank:
            raise ASMRuntimeError("SCAT requires src and dst to have the same rank", location=location, rewrite_rule="SCAT")
        if len(ind.shape) != 2 or ind.shape[0] != rank or ind.shape[1] != 2:
            raise ASMRuntimeError("SCAT indices must have shape [rank, 2]", location=location, rewrite_rule="SCAT")

        pairs = ind.data.reshape((ind.shape[0], ind.shape[1]))
        slices: List[Tuple[int, int]] = []

        for axis in range(rank):
            lo_raw = self._expect_int(pairs[axis, 0], "SCAT", location)
            hi_raw = self._expect_int(pairs[axis, 1], "SCAT", location)
            dim_len = dst.shape[axis]
            lo = self._resolve_tensor_index(lo_raw, dim_len, "SCAT", location)
            hi = self._resolve_tensor_index(hi_raw, dim_len, "SCAT", location)
            if lo > hi:
                raise ASMRuntimeError("SCAT expects lo <= hi for each dimension", location=location, rewrite_rule="SCAT")
            span = hi - lo + 1
            if span != src.shape[axis]:
                raise ASMRuntimeError("SCAT source shape must match index span", location=location, rewrite_rule="SCAT")
            slices.append((lo - 1, hi))

        out_data = dst.data.copy()
        dst_view = out_data.reshape(tuple(dst.shape))
        src_view = src.data.reshape(tuple(src.shape))
        dst_view[tuple(slice(start, end) for start, end in slices)] = src_view
        return Value(TYPE_TNS, Tensor(shape=list(dst.shape), data=out_data))

    def _parallel(
        self,
        interpreter: "Interpreter",
        args: List[Value],
        __: List[Expression],
        ___: Environment,
        location: SourceLocation,
    ) -> Value:
        """PARALLEL(TNS: functions):INT

        Execute each element of the `functions` tensor in parallel. Each
        element must be a `FUNC` value. Wait for all to complete and return
        integer 0 on success. If any element is not a function or any
        invocation raises, an ASMRuntimeError is raised.
        """
        # Support two forms:
        #  - PARALLEL(TNS: functions)
        #  - PARALLEL(FUNC, FUNC, ...)
        elems: List[Any]
        if len(args) == 1 and args[0].type == TYPE_TNS:
            tensor = args[0].value
            data = tensor.data
            flat = data.ravel()
            elems = [flat[i] for i in range(flat.size)]
        else:
            elems = list(args)

        n = len(elems)
        results: List[Optional[Value]] = [None] * n
        errors: List[Optional[BaseException]] = [None] * n

        def worker(idx: int, elem: Any) -> None:
            try:
                if not isinstance(elem, Value) or elem.type != TYPE_FUNC:
                    raise ASMRuntimeError("PARALLEL expects functions (either a tensor of FUNC or FUNC arguments)", location=location, rewrite_rule="PARALLEL")
                func = elem.value
                # Invoke with no args and the provided environment as closure
                res = interpreter._invoke_function_object(func, [], {}, location, ___)
                results[idx] = res
            except BaseException as exc:
                errors[idx] = exc

        threads: List[threading.Thread] = []
        for i in range(n):
            t = threading.Thread(target=worker, args=(i, elems[i]))
            t.start()
            threads.append(t)

        for t in threads:
            t.join()

        # Propagate first error if any
        for err in errors:
            if err is not None:
                if isinstance(err, ASMRuntimeError):
                    raise err
                raise ASMRuntimeError(f"PARALLEL worker failed: {err}", location=location, rewrite_rule="PARALLEL")

        return Value(TYPE_INT, 0)

    def _convolve(
        self,
        interpreter: "Interpreter",
        args: List[Value],
        arg_nodes: List[Expression],
        ___: Environment,
        location: SourceLocation,
    ) -> Value:
        """CONV(TNS: x, TNS: kernel, ...kwargs):TNS

        Extended CONV: preserves original N-D convolution semantics when called
        as CONV(x, kernel). Additionally, when called with keyword args
        matching the 2D helper (`stride_w`, `stride_h`, `pad_w`, `pad_h`,
        `bias`) and when the input is a 3-D WHC tensor and the kernel is a
        4-D tensor `[kw,kh,in_c,out_c]`, perform a multi-output 2D
        convolution with optional stride/padding/bias similar to lib/cnn.asmln
        `CONV2D`.
        """

        x = self._expect_tns(args[0], "CONV", location)
        kernel = self._expect_tns(args[1], "CONV", location)

        # Parse optional keyword arguments supplied as named call arguments.
        stride_w = 1
        stride_h = 1
        pad_w = 0
        pad_h = 0
        bias_val: Optional[Value] = None

        for idx, node in enumerate(arg_nodes):
            # Only consider named arguments beyond the first two positional args.
            if idx < 2:
                continue
            name = None
            if isinstance(node, CallArgument):
                name = node.name
            # If name present, map it to the provided value.
            if name is not None:
                val = args[idx]
                if name == "stride_w":
                    stride_w = self._expect_int(val, "CONV", location)
                elif name == "stride_h":
                    stride_h = self._expect_int(val, "CONV", location)
                elif name == "pad_w":
                    pad_w = self._expect_int(val, "CONV", location)
                elif name == "pad_h":
                    pad_h = self._expect_int(val, "CONV", location)
                elif name == "bias":
                    bias_val = val
                else:
                    raise ASMRuntimeError(f"CONV: unknown keyword argument '{name}'", location=location, rewrite_rule="CONV")

        # If caller provided CONV2D-style kwargs and this looks like a WHC
        # input with a 4-D kernel [kw,kh,in_c,out_c], call the multi-output
        # 2-D convolution path (stride/pad/bias supported). Otherwise fall
        # back to the original N-D behavior.
        if len(x.shape) == 3 and len(kernel.shape) == 4:
            in_w, in_h, in_c = x.shape
            kw, kh, k_in_c, out_c = kernel.shape
            if k_in_c != in_c:
                raise ASMRuntimeError("CONV kernel input channels do not match input tensor channels", location=location, rewrite_rule="CONV")

            # Determine numeric output type
            def _uniform_numeric_type(t: Tensor, which: str) -> str:
                if t.data.size == 0:
                    raise ASMRuntimeError(f"CONV does not support empty {which} tensors", location=location, rewrite_rule="CONV")
                first = next(iter(t.data.flat)).type
                if first not in (TYPE_INT, TYPE_FLT):
                    raise ASMRuntimeError("CONV expects INT or FLT tensor elements", location=location, rewrite_rule="CONV")
                if any(v.type != first for v in t.data.flat):
                    raise ASMRuntimeError("CONV does not allow mixed element types within a tensor", location=location, rewrite_rule="CONV")
                return first

            x_type = _uniform_numeric_type(x, "input")
            k_type = _uniform_numeric_type(kernel, "kernel")
            out_type = TYPE_INT if (x_type == TYPE_INT and k_type == TYPE_INT) else TYPE_FLT

            # Zero-padding to match CONV2D helper behavior.
            p_w = in_w + (pad_w * 2)
            p_h = in_h + (pad_h * 2)

            out_w = ( (p_w - kw) // stride_w ) + 1
            out_h = ( (p_h - kh) // stride_h ) + 1
            if out_w < 1 or out_h < 1:
                raise ASMRuntimeError("CONV produced non-positive output dimensions", location=location, rewrite_rule="CONV")

            # Prepare padded input array (object dtype of Values)
            x_arr = x.data.reshape(tuple(x.shape))
            padded = np.empty((p_w, p_h, in_c), dtype=object)
            # Fill with zero pad (match lib/cnn.asmln PAD2D behavior used there)
            pad_zero = Value(TYPE_INT, 0) if out_type == TYPE_INT else Value(TYPE_FLT, 0.0)
            for px in range(p_w):
                for py in range(p_h):
                    for pc in range(in_c):
                        # Map back to original coordinates
                        ox = px - pad_w
                        oy = py - pad_h
                        if 0 <= ox < in_w and 0 <= oy < in_h:
                            padded[px, py, pc] = x_arr[ox, oy, pc]
                        else:
                            padded[px, py, pc] = pad_zero

            # Prepare kernel and bias arrays
            k_arr = kernel.data.reshape(tuple(kernel.shape))
            bias_tns: Optional[Tensor] = None
            use_bias = False
            if bias_val is not None:
                bias_tns = self._expect_tns(bias_val, "CONV", location)
                if bias_tns.data.size == out_c:
                    use_bias = True

            out_buf = np.empty(out_w * out_h * out_c, dtype=object)
            idx_out = 0
            for oc in range(out_c):
                b = 0
                if use_bias:
                    bv = bias_tns.data.flat[oc]
                    b = int(bv.value) if out_type == TYPE_INT else float(bv.value)
                for oy in range(out_h):
                    for ox in range(out_w):
                        acc_int = 0
                        acc_flt = 0.0
                        base_x = ox * stride_w
                        base_y = oy * stride_h
                        for ic in range(in_c):
                            for ky in range(kh):
                                for kx in range(kw):
                                    px = base_x + kx
                                    py = base_y + ky
                                    xv: Value = padded[px, py, ic]
                                    kv: Value = k_arr[kx, ky, ic, oc]
                                    if out_type == TYPE_INT:
                                        acc_int += int(xv.value) * int(kv.value)
                                    else:
                                        ax = float(xv.value) if xv.type == TYPE_FLT else float(int(xv.value))
                                        ak = float(kv.value) if kv.type == TYPE_FLT else float(int(kv.value))
                                        acc_flt += ax * ak
                        if out_type == TYPE_INT:
                            val = acc_int + (int(b) if use_bias else 0)
                            out_buf[idx_out] = Value(TYPE_INT, val)
                        else:
                            val = acc_flt + (float(b) if use_bias else 0.0)
                            out_buf[idx_out] = Value(TYPE_FLT, float(val))
                        idx_out += 1

            return Value(TYPE_TNS, Tensor(shape=[out_w, out_h, out_c], data=out_buf))

        # Fallback: preserve original N-D behavior (replicate boundaries,
        # odd kernel dims, same-shape output). This is the original path
        # (unchanged).
        if len(x.shape) != len(kernel.shape):
            raise ASMRuntimeError(
                "CONV requires input and kernel tensors with the same rank",
                location=location,
                rewrite_rule="CONV",
            )
        if any((d % 2) == 0 for d in kernel.shape):
            raise ASMRuntimeError(
                "CONV requires odd kernel dimensions",
                location=location,
                rewrite_rule="CONV",
            )

        def _uniform_numeric_type(t: Tensor, which: str) -> str:
            if t.data.size == 0:
                raise ASMRuntimeError(
                    f"CONV does not support empty {which} tensors",
                    location=location,
                    rewrite_rule="CONV",
                )
            first = next(iter(t.data.flat)).type
            if first not in (TYPE_INT, TYPE_FLT):
                raise ASMRuntimeError(
                    "CONV expects INT or FLT tensor elements",
                    location=location,
                    rewrite_rule="CONV",
                )
            if any(v.type != first for v in t.data.flat):
                raise ASMRuntimeError(
                    "CONV does not allow mixed element types within a tensor",
                    location=location,
                    rewrite_rule="CONV",
                )
            return first

        x_type = _uniform_numeric_type(x, "input")
        k_type = _uniform_numeric_type(kernel, "kernel")
        out_type = TYPE_INT if (x_type == TYPE_INT and k_type == TYPE_INT) else TYPE_FLT

        rank = len(x.shape)
        centers = [d // 2 for d in kernel.shape]  # 0-based

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
            x_arr = x.data.reshape(tuple(x.shape))
            k_arr = kernel.data.reshape(tuple(kernel.shape))
            out = np.empty(x.data.size, dtype=object)
            out_i = 0
            for out_pos in np.ndindex(*x.shape):
                acc_int = 0
                acc_flt = 0.0
                for k_pos in np.ndindex(*kernel.shape):
                    in_pos: List[int] = []
                    for axis in range(rank):
                        offset = k_pos[axis] - centers[axis]
                        coord = out_pos[axis] + offset
                        if coord < 0:
                            coord = 0
                        elif coord >= x.shape[axis]:
                            coord = x.shape[axis] - 1
                        in_pos.append(coord)

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

    # --- thread builtins ---
    def _resolve_thr_from_value(self, interpreter: "Interpreter", val: Value, rule: str, location: SourceLocation):
        val = interpreter._deref_value(val, location=location, rule=rule)
        if val.type != TYPE_THR:
            raise ASMRuntimeError(f"{rule} expects THR value", location=location, rewrite_rule=rule)
        ctrl = val.value
        return ctrl

    def _stop_thr(self, interpreter: "Interpreter", args: List[Value], __: List[Expression], ___: Environment, location: SourceLocation) -> Value:
        ctrl = self._resolve_thr_from_value(interpreter, args[0], "STOP", location)
        thr = ctrl.get("thread")
        if thr is None:
            raise ASMRuntimeError("Invalid THR", location=location, rewrite_rule="STOP")
        # Cooperative stop: mark as stopping/stopped and allow worker to exit at
        # the next statement boundary.
        ctrl["stop"] = True
        ctrl["finished"] = True
        ctrl["paused"] = False
        try:
            ctrl["pause_event"].set()
        except Exception:
            pass
        ctrl["state"] = "stopped"
        return Value(TYPE_THR, ctrl)

    def _await_thr(self, interpreter: "Interpreter", args: List[Value], __: List[Expression], ___: Environment, location: SourceLocation) -> Value:
        ctrl = self._resolve_thr_from_value(interpreter, args[0], "AWAIT", location)
        thr = ctrl.get("thread")
        if thr is None:
            raise ASMRuntimeError("Invalid THR", location=location, rewrite_rule="AWAIT")
        thr.join()
        return Value(TYPE_THR, ctrl)

    def _pause_thr(self, interpreter: "Interpreter", args: List[Value], __: List[Expression], ___: Environment, location: SourceLocation) -> Value:
        ctrl = self._resolve_thr_from_value(interpreter, args[0], "PAUSE", location)
        seconds = -1.0
        if len(args) > 1:
            seconds = self._expect_flt(args[1], "PAUSE", location)
        if ctrl.get("paused"):
            raise ASMRuntimeError("Thread already paused", location=location, rewrite_rule="PAUSE")
        # Mark paused; threads need to check pause_event to actually pause.
        ctrl["paused"] = True
        ctrl["pause_event"].clear()
        ctrl["state"] = "paused"
        if seconds is not None and seconds >= 0:
            def _delayed_resume():
                time.sleep(seconds)
                ctrl["paused"] = False
                ctrl["pause_event"].set()
                ctrl["state"] = "running"

            threading.Thread(target=_delayed_resume, daemon=True).start()
        return Value(TYPE_THR, ctrl)

    def _resume_thr(self, interpreter: "Interpreter", args: List[Value], __: List[Expression], ___: Environment, location: SourceLocation) -> Value:
        ctrl = self._resolve_thr_from_value(interpreter, args[0], "RESUME", location)
        if not ctrl.get("paused"):
            raise ASMRuntimeError("Thread is not paused", location=location, rewrite_rule="RESUME")
        ctrl["paused"] = False
        ctrl["pause_event"].set()
        ctrl["state"] = "running"
        return Value(TYPE_THR, ctrl)

    def _paused_thr(self, interpreter: "Interpreter", args: List[Value], __: List[Expression], ___: Environment, location: SourceLocation) -> Value:
        ctrl = self._resolve_thr_from_value(interpreter, args[0], "PAUSED", location)
        return Value(TYPE_INT, 1 if ctrl.get("paused") else 0)

    def _restart_thr(self, interpreter: "Interpreter", args: List[Value], __: List[Expression], ___: Environment, location: SourceLocation) -> Value:
        ctrl = self._resolve_thr_from_value(interpreter, args[0], "RESTART", location)
        # Not fully reinitializing environment; best-effort: if original env available, restart
        env = ctrl.get("env")
        block = ctrl.get("block")
        if env is None or block is None:
            raise ASMRuntimeError("Cannot restart THR", location=location, rewrite_rule="RESTART")
        # Reset finished flag and spawn new thread
        ctrl["stop"] = False
        ctrl["paused"] = False
        try:
            ctrl["pause_event"].set()
        except Exception:
            pass
        ctrl["finished"] = False
        ctrl["state"] = "running"

        def _worker_restart():
            frame = interpreter._new_frame("<thr>", env, location)
            interpreter._thread_ctrls[threading.get_ident()] = ctrl
            interpreter.call_stack.append(frame)
            try:
                interpreter._execute_block(block.statements, env)
            finally:
                try:
                    interpreter.call_stack.pop()
                except Exception:
                    pass
                try:
                    tid = threading.get_ident()
                    if tid in interpreter._thread_ctrls:
                        del interpreter._thread_ctrls[tid]
                except Exception:
                    pass
                ctrl["finished"] = True
                ctrl["state"] = "finished" if not ctrl.get("stop") else "stopped"

        t = threading.Thread(target=_worker_restart, daemon=True)
        ctrl["thread"] = t
        t.start()
        return Value(TYPE_THR, ctrl)


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
        normalized_filename = filename if filename == "<repl>" else os.path.abspath(filename)
        self.filename = normalized_filename
        self.entry_filename = normalized_filename
        self.verbose = verbose
        self.services = services or build_default_services()
        self.type_registry: TypeRegistry = self.services.type_registry
        self.hook_registry: HookRegistry = self.services.hook_registry
        self.input_provider = input_provider or (lambda: input(">>> "))
        self.output_sink = output_sink or (lambda text: print(text))
        self.builtins = Builtins()
        # Pending async thread starts created during expression evaluation.
        # ASYNC expressions create a thread controller but defer starting the
        # worker until the enclosing call expression completes so that
        # operators like PAUSE/STOP can act on the returned THR before it runs.
        self._pending_async_starts: List[Dict[str, Any]] = []

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

        if not self.type_registry.has(TYPE_MAP):
            self.type_registry.register(
                TypeSpec(
                    name=TYPE_MAP,
                    printable=True,
                    condition_int=lambda ctx, v: 1 if getattr(v.value, "data", {}) else 0,
                    to_str=lambda ctx, v: "<map>",
                ),
                seal=True,
            )

        if not self.type_registry.has(TYPE_THR):
            # THR is truthy while running/paused, false when finished
            def _thr_to_str(ctx: TypeContext, v: Value) -> str:
                obj = v.value
                state = getattr(obj, "state", "finished")
                return f"<thr {state}>"

            def _thr_cond(ctx: TypeContext, v: Value) -> int:
                obj = v.value
                # finished -> 0, otherwise 1
                return 0 if getattr(obj, "finished", True) else 1

            self.type_registry.register(
                TypeSpec(
                    name=TYPE_THR,
                    printable=True,
                    condition_int=_thr_cond,
                    to_str=_thr_to_str,
                ),
                seal=True,
            )

        if not self.type_registry.has(TYPE_FUNC):
            def _func_to_str(ctx: TypeContext, v: Value) -> str:
                name = getattr(v.value, "name", None)
                return f"<func {name}>" if name else "<func>"

            self.type_registry.register(
                TypeSpec(
                    name=TYPE_FUNC,
                    printable=True,
                    condition_int=lambda ctx, v: 1,
                    to_str=_func_to_str,
                    equals=lambda ctx, a, b: a.value is b.value,
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

        # Register thread-related builtins
        self.builtins.register_extension_operator(name="STOP", min_args=1, max_args=1, impl=self.builtins._stop_thr)
        self.builtins.register_extension_operator(name="AWAIT", min_args=1, max_args=1, impl=self.builtins._await_thr)
        self.builtins.register_extension_operator(name="PAUSE", min_args=1, max_args=2, impl=self.builtins._pause_thr)
        self.builtins.register_extension_operator(name="RESUME", min_args=1, max_args=1, impl=self.builtins._resume_thr)
        self.builtins.register_extension_operator(name="PAUSED", min_args=1, max_args=1, impl=self.builtins._paused_thr)
        self.builtins.register_extension_operator(name="RESTART", min_args=1, max_args=1, impl=self.builtins._restart_thr)

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

        # Threading / THR management
        # Map from thread id (symbol name) to a small controller object
        self._thr_lock = threading.Lock()
        self._thrs: Dict[str, Any] = {}

        # Cache for resolving unqualified function calls inside module frames.
        # Keyed by (frame_name, callee_name, functions_version).
        self._functions_version: int = 0
        self._call_resolution_cache: Dict[Tuple[str, str, int], Optional[str]] = {}

        # Monotonic counter for anonymous lambda names (for logs/tracebacks only).
        self._lambda_counter: int = 0
        # Map OS thread id -> controller dict for cooperative pause/inspect
        self._thread_ctrls: Dict[int, Any] = {}

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

    def _expect_func(self, value: "Value", rule: str, location: SourceLocation) -> Function:
        value = self._deref_value(value, location=location, rule=rule)
        if value.type != TYPE_FUNC:
            raise ASMRuntimeError(f"{rule} expects function value", location=location, rewrite_rule=rule)
        func_obj = value.value
        if not isinstance(func_obj, Function):
            raise ASMRuntimeError("Invalid function value", location=location, rewrite_rule=rule)
        return func_obj

    # ---- Pointer helpers ----
    def _resolve_pointer_target(
        self, name: str, env: Environment, location: SourceLocation, *, rule: str = "POINTER"
    ) -> Tuple[Environment, str, Value]:
        current_env = env
        current_name = name
        hops = 0
        while True:
            target_val = current_env.get_optional(current_name)
            if target_val is None:
                raise ASMRuntimeError(
                    f"Pointer target '{current_name}' is undefined",
                    location=location,
                    rewrite_rule=rule,
                )
            if not isinstance(target_val.value, PointerRef):
                return current_env, current_name, target_val
            hops += 1
            if hops > 128:
                raise ASMRuntimeError("Pointer cycle detected", location=location, rewrite_rule=rule)
            ptr = target_val.value
            current_env = ptr.env
            current_name = ptr.name

    def _make_pointer_value(self, name: str, env: Environment, location: SourceLocation) -> Value:
        target_env = env._find_env(name)
        if target_env is None:
            raise ASMRuntimeError(
                f"Pointer target '{name}' is undefined",
                location=location,
                rewrite_rule="POINTER",
            )
        # Do not allow creating a new pointer that targets a frozen or
        # permanently-frozen identifier. Existing pointers or frozen
        # identifiers may remain frozen, but no new aliasing may be created
        # until the identifier is thawed.
        if name in target_env.frozen and name not in target_env.permafrozen:
            raise ASMRuntimeError(
                f"Cannot create pointer to frozen identifier '{name}'",
                location=location,
                rewrite_rule="POINTER",
            )
        elif name in target_env.permafrozen:
            raise ASMRuntimeError(
                f"Cannot create pointer to permanently-frozen identifier '{name}'",
                location=location,
                rewrite_rule="POINTER",
            )
        resolved_env, resolved_name, resolved_val = self._resolve_pointer_target(name, target_env, location)
        return Value(resolved_val.type, PointerRef(env=resolved_env, name=resolved_name))

    def _deref_value(
        self,
        value: Value,
        *,
        location: Optional[SourceLocation] = None,
        rule: Optional[str] = None,
    ) -> Value:
        current = value
        hops = 0
        while isinstance(current.value, PointerRef):
            hops += 1
            if hops > 128:
                raise ASMRuntimeError("Pointer cycle detected", location=location, rewrite_rule=rule or "POINTER")
            ptr = current.value
            target_val = ptr.env.get_optional(ptr.name)
            if target_val is None:
                raise ASMRuntimeError(
                    f"Pointer target '{ptr.name}' is undefined",
                    location=location,
                    rewrite_rule=rule or "POINTER",
                )
            current = target_val
        return current

    def _assign_pointer_target(
        self, ptr: PointerRef, new_value: Value, *, location: Optional[SourceLocation], rule: str
    ) -> None:
        try:
            ptr.env.set(ptr.name, new_value)
        except ASMRuntimeError as err:
            if err.location is None:
                err.location = location
            if err.rewrite_rule is None:
                err.rewrite_rule = rule
            raise
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
        n_statements = len(statements)
        emit_event = self._emit_event
        log_step = self._log_step
        eval_expr = self._evaluate_expression
        execute_stmt = self._execute_statement

        frame: Frame = self.call_stack[-1]
        gotopoints: Dict[int, int] = frame.gotopoints
        while i < n_statements:
            statement = statements[i]
            emit_event("before_statement", self, statement, env)
            # Cooperative pause: if the current OS thread has an associated
            # controller and it is paused, block until resumed.
            try:
                tid = threading.get_ident()
                ctrl = self._thread_ctrls.get(tid)
                if ctrl is not None:
                    # Wait while paused; pause_event is set when running.
                    while ctrl.get("paused"):
                        ctrl.get("pause_event").wait()
                    # Cooperative stop: exit at statement boundary.
                    if ctrl.get("stop"):
                        return
            except Exception:
                pass
            # TryStatement is handled directly here so that catch binding
            # lifetime does not escape the following statement sequencing.
            if type(statement) is TryStatement:
                log_step(rule=statement.__class__.__name__, location=statement.location)
                try:
                    self._execute_block(statement.try_block.statements, env)
                except ASMRuntimeError as err:
                    # Optionally bind symbol and run catch block
                    if statement.catch_symbol is not None:
                        name = statement.catch_symbol
                        # Find existing binding (if any) and record it for restore
                        found_env = env._find_env(name)
                        prev_val = None
                        had_prev = False
                        if found_env is not None:
                            prev_val = found_env.values.get(name)
                            had_prev = True
                        # Shadow by inserting into the current environment's values
                        env.values[name] = Value(TYPE_STR, err.message)
                        try:
                            self._execute_block(statement.catch_block.statements, env)
                        finally:
                            # Restore previous binding if present, otherwise remove the shadow
                            try:
                                if had_prev and prev_val is not None:
                                    # Restore into the environment where it was found
                                    assert found_env is not None
                                    found_env.values[name] = prev_val
                                else:
                                    # Remove the shadow binding from the current env
                                    if name in env.values:
                                        del env.values[name]
                            except ASMRuntimeError:
                                raise
                    else:
                        # No symbol to bind: run catch block without binding
                        self._execute_block(statement.catch_block.statements, env)
                except Exception as exc:
                    # Non-ASMRuntimeError should propagate as usual
                    raise
                i += 1
                emit_event("after_statement", self, statement, env)
                continue
            if type(statement) is GotopointStatement:
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
        statement_type = type(statement)
        if statement_type is Declaration:
            # Record a type declaration without creating a value. Do not
            # introduce a binding; reads will still raise until the name is
            # assigned. If a value already exists in the same environment,
            # ensure the types match.
            if statement.name in self.functions:
                raise ASMRuntimeError(
                    f"Identifier '{statement.name}' already bound as function",
                    location=statement.location,
                    rewrite_rule="ASSIGN",
                )
            env_found = env._find_env(statement.name)
            if env_found is not None:
                existing = env_found.values.get(statement.name)
                if existing is not None and existing.type != statement.declared_type:
                    raise ASMRuntimeError(
                        f"Type mismatch for '{statement.name}': previously declared as {existing.type}",
                        location=statement.location,
                        rewrite_rule="ASSIGN",
                    )
            env.declared[statement.name] = statement.declared_type
            return
        if statement_type is Assignment:
            if statement.target in self.functions:
                raise ASMRuntimeError(
                    f"Identifier '{statement.target}' already bound as function", location=statement.location, rewrite_rule="ASSIGN"
                )
            value = self._evaluate_expression(statement.expression, env)
            env.set(statement.target, value, declared_type=statement.declared_type)
            return
        if statement_type is TensorSetStatement:
            base_expr, index_nodes = self._gather_index_chain(statement.target)
            if type(base_expr) is not Identifier:
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
            base_val = self._deref_value(base_val, location=statement.location, rule="ASSIGN")
            # MAP assignment path: support multi-key assignment like m<k1,k2> = v
            if base_val.type == TYPE_MAP:
                assert isinstance(base_val.value, Map)
                eval_expr = self._evaluate_expression
                # Traverse/create nested maps for all but the final key
                current_map_val = base_val
                for idx_node in index_nodes[:-1]:
                    key_val = eval_expr(idx_node, env)
                    if key_val.type not in (TYPE_INT, TYPE_FLT, TYPE_STR):
                        raise ASMRuntimeError("Map keys must be INT, FLT, or STR", location=statement.location, rewrite_rule="ASSIGN")
                    key = (key_val.type, key_val.value)
                    # If key absent, create nested map
                    if key not in current_map_val.value.data:
                        new_map = Map()
                        current_map_val.value.data[key] = Value(TYPE_MAP, new_map)
                    next_val = current_map_val.value.data[key]
                    if next_val.type != TYPE_MAP:
                        raise ASMRuntimeError("Cannot index into non-map value", location=statement.location, rewrite_rule="ASSIGN")
                    current_map_val = next_val
                # Final key
                final_key_node = index_nodes[-1]
                final_key_val = eval_expr(final_key_node, env)
                if final_key_val.type not in (TYPE_INT, TYPE_FLT, TYPE_STR):
                    raise ASMRuntimeError("Map keys must be INT, FLT, or STR", location=statement.location, rewrite_rule="ASSIGN")
                final_key = (final_key_val.type, final_key_val.value)
                new_value = self._evaluate_expression(statement.value, env)
                existing = current_map_val.value.data.get(final_key)
                if existing is not None and existing.type != new_value.type:
                    raise ASMRuntimeError(
                        f"Type mismatch for map key assignment: existing {existing.type} vs new {new_value.type}",
                        location=statement.location,
                        rewrite_rule="ASSIGN",
                    )
                current_map_val.value.data[final_key] = new_value
                return

            # Tensor assignment path (existing semantics)
            if base_val.type != TYPE_TNS:
                raise ASMRuntimeError(
                    "Indexed assignment requires a tensor or map base",
                    location=statement.location,
                    rewrite_rule="ASSIGN",
                )
            assert isinstance(base_val.value, Tensor)
            eval_expr = self._evaluate_expression
            expect_int = self._expect_int
            # Check if any index is a Range (slice). If none, mutate a single
            # element as before. If ranges present, perform a slice assignment.
            RangeT = Range
            StarT = Star
            has_range = any((type(n) is RangeT) or (type(n) is StarT) for n in index_nodes)
            if not has_range:
                indices: List[int] = []
                indices_append = indices.append
                for node in index_nodes:
                    indices_append(expect_int(eval_expr(node, env), "ASSIGN", statement.location))
                new_value = self._evaluate_expression(statement.value, env)
                self._mutate_tensor_value(base_val.value, indices, new_value, statement.location)
                return

            # Slice assignment path
            arr = base_val.value.data.reshape(tuple(base_val.value.shape))
            indexers: List[object] = []
            for dim, node in enumerate(index_nodes):
                node_type = type(node)
                if node_type is Range:
                    lo_val = expect_int(eval_expr(node.lo, env), "ASSIGN", statement.location)
                    hi_val = expect_int(eval_expr(node.hi, env), "ASSIGN", statement.location)
                    lo_res = self._resolve_tensor_index(lo_val, base_val.value.shape[dim], "ASSIGN", statement.location)
                    hi_res = self._resolve_tensor_index(hi_val, base_val.value.shape[dim], "ASSIGN", statement.location)
                    indexers.append(slice(lo_res - 1, hi_res))
                elif node_type is Star:
                    # Full-dimension slice `*` selects the entire axis
                    indexers.append(slice(None, None))
                else:
                    raw = expect_int(eval_expr(node, env), "ASSIGN", statement.location)
                    idx_res = self._resolve_tensor_index(raw, base_val.value.shape[dim], "ASSIGN", statement.location)
                    indexers.append(idx_res - 1)

            sel = arr[tuple(indexers)]
            # Evaluate RHS and ensure it's a tensor of matching shape
            new_value = self._evaluate_expression(statement.value, env)
            if new_value.type != TYPE_TNS:
                raise ASMRuntimeError("Slice assignment requires tensor value", location=statement.location, rewrite_rule="ASSIGN")
            assert isinstance(new_value.value, Tensor)
            sel_shape = list(sel.shape) if isinstance(sel, np.ndarray) else list(np.array(sel, dtype=object).shape)
            if sel_shape != new_value.value.shape:
                raise ASMRuntimeError("Slice assignment shape mismatch", location=statement.location, rewrite_rule="ASSIGN")

            # Elementwise type check then mutate in place
            sel_view = sel if isinstance(sel, np.ndarray) else np.array(sel, dtype=object)
            sel_flat = sel_view.ravel()
            new_flat = new_value.value.data.ravel()
            if sel_flat.size != new_flat.size:
                raise ASMRuntimeError("Slice assignment size mismatch", location=statement.location, rewrite_rule="ASSIGN")
            # Verify element type compatibility
            for i in range(sel_flat.size):
                cur = sel_flat[i]
                newv = new_flat[i]
                if cur.type != newv.type:
                    raise ASMRuntimeError("Tensor element type mismatch", location=statement.location, rewrite_rule="ASSIGN")
            # Perform mutation on the original array view
            # Use flat indexing into the original view to preserve object identity
            target_view = arr[tuple(indexers)]
            target_flat = target_view.ravel()
            # Assign element-wise
            for i in range(new_flat.size):
                target_flat[i] = new_flat[i]
            return
        if statement_type is ExpressionStatement:
            self._evaluate_expression(statement.expression, env)
            return
        if statement_type is IfStatement:
            self._execute_if(statement, env)
            return
        if statement_type is WhileStatement:
            self._execute_while(statement, env)
            return
        if statement_type is ForStatement:
            self._execute_for(statement, env)
            return
        if statement_type is ParForStatement:
            self._execute_parfor(statement, env)
            return
        if statement_type is FuncDef:
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
        if statement_type is PopStatement:
            frame: Frame = self.call_stack[-1]
            if frame.name == "<top-level>":
                raise ASMRuntimeError("POP outside of function", location=statement.location, rewrite_rule="POP")
            # Expect identifier expression to delete a symbol
            expr = statement.expression
            if type(expr) is not Identifier:
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
        if statement_type is ReturnStatement:
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
                elif fn.return_type == TYPE_FUNC:
                    raise ASMRuntimeError(
                        "FUNC functions must RETURN a function value",
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
        if statement_type is BreakStatement:
            count_val = self._evaluate_expression(statement.expression, env)
            count = self._expect_int(count_val, "BREAK", statement.location)
            if count <= 0:
                raise ASMRuntimeError("BREAK count must be > 0", location=statement.location, rewrite_rule="BREAK")
            raise BreakSignal(count)
        if statement_type is ContinueStatement:
            # Signal to the innermost loop to skip to next iteration.
            raise ContinueSignal()
        if statement_type is GotoStatement:
            target = self._evaluate_expression(statement.expression, env)
            raise JumpSignal(target)
        if statement_type.__name__ == "ThrStatement":
            # Create a named THR controller and start execution in background
            # Bind the symbol in current env to the THR value.
            symbol = statement.symbol  # type: ignore[attr-defined]
            block = statement.block  # type: ignore[attr-defined]
            loc = statement.location

            ctrl = {
                "thread": None,
                "paused": False,
                "pause_event": threading.Event(),
                "finished": False,
                "stop": False,
                "state": "running",
                "env": env,
                "block": block,
            }
            # controller starts unpaused
            ctrl["pause_event"].set()

            def _thr_worker():
                frame = self._new_frame(f"<thr:{symbol}>", env, loc)
                # register ctrl for this OS thread
                self._thread_ctrls[threading.get_ident()] = ctrl
                self.call_stack.append(frame)
                try:
                    self._emit_event("thr_start", self, frame, env, symbol)
                    try:
                        self._execute_block(block.statements, env)
                    except Exception as exc:
                        try:
                            self._emit_event("on_error", self, exc)
                        except Exception:
                            pass
                finally:
                    try:
                        self.call_stack.pop()
                    except Exception:
                        pass
                    ctrl["finished"] = True
                    ctrl["state"] = "finished" if not ctrl.get("stop") else "stopped"
                    try:
                        self._emit_event("thr_end", self, frame, env, symbol)
                    except Exception:
                        pass
                    # unregister ctrl
                    try:
                        tid = threading.get_ident()
                        if tid in self._thread_ctrls:
                            del self._thread_ctrls[tid]
                    except Exception:
                        pass

            t = threading.Thread(target=_thr_worker, daemon=True, name=f"asm_thr_{symbol}_{self.frame_counter}")
            ctrl["thread"] = t
            with self._thr_lock:
                if symbol in self._thrs:
                    raise ASMRuntimeError(f"THR symbol '{symbol}' already exists", location=statement.location, rewrite_rule="THR")
                self._thrs[symbol] = ctrl
            # Bind symbol to THR value in environment
            env.set(symbol, Value(TYPE_THR, ctrl), declared_type="THR")
            t.start()
            return
        if statement_type is AsyncStatement:
            # Execute the block synchronously inside a background thread
            block = statement.block
            loc = statement.location

            def _async_worker() -> None:
                frame = self._new_frame("<async>", env, loc)
                self.call_stack.append(frame)
                try:
                    self._emit_event("async_start", self, frame, env)
                    try:
                        self._execute_block(block.statements, env)
                    except Exception as exc:
                        # Notify hooks about the error so it can be recorded/handled.
                        try:
                            self._emit_event("on_error", self, exc)
                        except Exception:
                            pass
                finally:
                    # Pop the async frame and emit end event.
                    try:
                        self.call_stack.pop()
                    except Exception:
                        pass
                    try:
                        self._emit_event("async_end", self, frame, env)
                    except Exception:
                        pass

            # For compatibility, ASYNC as a statement fires and returns a THR-like
            # controller which is ignored by statement context. Create a controller
            # and start a thread.
            ctrl = {
                "thread": None,
                "paused": False,
                "pause_event": threading.Event(),
                "finished": False,
                "stop": False,
                "state": "running",
                "block": block,
                "env": env,
            }

            # controller starts unpaused
            ctrl["pause_event"].set()

            def _async_worker_with_ctrl():
                frame = self._new_frame("<async>", env, loc)
                # register ctrl for this OS thread
                self._thread_ctrls[threading.get_ident()] = ctrl
                self.call_stack.append(frame)
                try:
                    self._emit_event("async_start", self, frame, env)
                    try:
                        self._execute_block(block.statements, env)
                    except Exception as exc:
                        try:
                            self._emit_event("on_error", self, exc)
                        except Exception:
                            pass
                finally:
                    try:
                        self.call_stack.pop()
                    except Exception:
                        pass
                    try:
                        self._emit_event("async_end", self, frame, env)
                    except Exception:
                        pass
                    ctrl["finished"] = True
                    ctrl["state"] = "finished" if not ctrl.get("stop") else "stopped"
                    # unregister ctrl
                    try:
                        tid = threading.get_ident()
                        if tid in self._thread_ctrls:
                            del self._thread_ctrls[tid]
                    except Exception:
                        pass

            t = threading.Thread(target=_async_worker_with_ctrl, daemon=True, name=f"asm_async_{self.frame_counter}")
            ctrl["thread"] = t
            t.start()
            return
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
        # Preserve any pre-existing binding for the counter so it can be
        # restored after the loop. The loop counter itself is loop-local
        # (not retained in the enclosing environment) but symbols created
        # inside the loop body are placed in the enclosing environment.
        prior = env.get_optional(statement.counter)
        had_prior = prior is not None
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
        # Restore or remove the loop counter binding so the counter does
        # not leak into the enclosing scope.
        try:
            if had_prior:
                assert prior is not None
                env.set(statement.counter, prior)
            else:
                # If the counter was newly created, delete it from the
                # nearest enclosing environment.
                try:
                    env.delete(statement.counter)
                except ASMRuntimeError:
                    # If deletion fails for any reason (e.g. frozen), leave
                    # the current binding in place; propagate nothing here.
                    pass
        except ASMRuntimeError:
            # If restoring the prior value or deleting failed, annotate
            # with loop location and re-raise.
            raise

    def _execute_parfor(self, statement: ParForStatement, env: Environment) -> None:
        eval_expr = self._evaluate_expression
        target_val = eval_expr(statement.target_expr, env)
        target = self._expect_int(target_val, "PARFOR", statement.location)

        threads: List[threading.Thread] = []
        errors: List[Exception] = []
        break_holder: List[BreakSignal] = []
        break_event = threading.Event()
        # Track child environments produced by each iteration so their
        # declared symbols (other than the counter) can be merged back
        # into the enclosing environment after iterations complete.
        child_envs: List[Environment] = []
        child_envs_lock = threading.Lock()

        # Capture the current (enclosing) frame name so workers can resolve
        # unqualified function names relative to the module that started the
        # PARFOR. We capture it here to avoid reading `self.call_stack` from
        # worker threads where the stack may differ.
        parent_frame_name = self.call_stack[-1].name if self.call_stack else "<top-level>"

        def _worker(iter_idx: int) -> None:
            # Create an isolated environment for this iteration by shallow-copying
            # the current environment's values so writes do not race with other
            # iterations. This keeps reads consistent but isolates mutation.
            child_env = Environment(parent=None, values=dict(env.values), frozen=set(env.frozen), permafrozen=set(env.permafrozen))
            # Set the loop counter (1-indexed)
            child_env.values[statement.counter] = Value(TYPE_INT, iter_idx)
            # Record this child environment for later merge.
            try:
                with child_envs_lock:
                    child_envs.append(child_env)
            except Exception:
                # Best-effort: if we cannot record the env, proceed anyway.
                pass
            # Use a frame name that preserves the enclosing frame's module
            # prefix so resolution of unqualified function calls (e.g.
            # `AUDIO_CLAMP`) inside the iteration will find module-local
            # functions such as `waveforms.AUDIO_CLAMP`.
            frame_name = f"{parent_frame_name}.<parfor>"
            frame = self._new_frame(frame_name, child_env, statement.location)
            self.call_stack.append(frame)
            try:
                try:
                    # If a BREAK already occurred in another iteration, skip work.
                    if break_event.is_set():
                        return
                    self._execute_block(statement.block.statements, child_env)
                except BreakSignal as bs:
                    # Signal termination of the whole PARFOR.
                    break_holder.append(bs)
                    break_event.set()
                    return
                except ContinueSignal:
                    # CONTINUE ends this iteration only.
                    return
                except Exception as exc:
                    # record exception for propagation after join
                    errors.append(exc)
            finally:
                try:
                    self.call_stack.pop()
                except Exception:
                    pass

        for i in range(1, target + 1):
            # If a BREAK has been signaled, stop starting further iterations.
            if break_event.is_set():
                break
            t = threading.Thread(target=lambda idx=i: _worker(idx), name=f"asm_parfor_{self.frame_counter}_{i}")
            threads.append(t)
            t.start()

        # Wait for all iterations to complete
        for t in threads:
            t.join()
        # Merge declared/modified symbols from child envs back into the
        # enclosing environment. Skip the loop counter which is iteration-
        # local. If multiple iterations wrote the same name, later writes
        # win (last-writer-wins). Merging happens before propagating any
        # BREAK or iteration errors so user-visible state is consistent.
        orig_keys = set(env.values.keys())
        for child in child_envs:
            for name, val in child.values.items():
                if name == statement.counter:
                    continue
                try:
                    if name in orig_keys:
                        env.set(name, val)
                    else:
                        env.set(name, val, declared_type=val.type)
                except ASMRuntimeError as err:
                    err.location = statement.location
                    raise

        # If a BREAK was raised in any iteration, propagate it to the caller
        if break_holder:
            # Re-raise the first BreakSignal so outer loops may handle it.
            raise break_holder[0]

        if errors:
            # Raise the first recorded exception as a runtime error
            exc = errors[0]
            loc = statement.location
            if isinstance(exc, ASMRuntimeError):
                raise exc
            raise ASMRuntimeError(f"Error in PARFOR iteration: {exc}", location=loc, rewrite_rule="PARFOR")

    def _condition_int(self, value: Value, location: Optional[SourceLocation]) -> int:
        value = self._deref_value(value, location=location, rule="COND")
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
        value = self._deref_value(value, location=location, rule=rule)
        if value.type != TYPE_INT:
            raise ASMRuntimeError(f"{rule} expects integer value", location=location, rewrite_rule=rule)
        return value.value  # type: ignore[return-value]

    def _tensor_total_size(self, shape: List[int]) -> int:
        size = 1
        for dim in shape:
            size *= dim
        return size

    def _tensor_truthy(self, tensor: Tensor) -> bool:
        # Iterate over a flattened view with a local binding for the
        # condition evaluator to reduce attribute lookups in tight loops.
        cond_int = self._condition_int
        flat = tensor.data.ravel()
        for i in range(flat.size):
            if cond_int(flat[i], None) != 0:
                return True
        return False

    def _values_equal(self, left: Value, right: Value) -> bool:
        left = self._deref_value(left)
        right = self._deref_value(right)
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
        # Use indexed loops over ravel() to avoid tuple allocations and
        # generator overhead in large tensors.
        l_flat = left.data.ravel()
        r_flat = right.data.ravel()
        n = l_flat.size
        for i in range(n):
            if not self._values_equal(l_flat[i], r_flat[i]):
                return False
        return True

    def _validate_tensor_shape(self, shape: List[int], rule: str, location: SourceLocation) -> None:
        if not shape:
            raise ASMRuntimeError("Tensor shape must have at least one dimension", location=location, rewrite_rule=rule)
        for dim in shape:
            if dim <= 0:
                raise ASMRuntimeError("Tensor dimensions must be positive", location=location, rewrite_rule=rule)

    def _resolve_tensor_index(self, raw: int, dim_len: int, rule: str, location: SourceLocation) -> int:
        """Map a 1-based (or negative-from-end) index into the current dimension."""
        if raw == 0:
            raise ASMRuntimeError("Tensor indices are 1-indexed", location=location, rewrite_rule=rule)
        idx = raw
        if raw < 0:
            idx = dim_len + raw + 1
        if idx <= 0 or idx > dim_len:
            raise ASMRuntimeError("Tensor index out of range", location=location, rewrite_rule=rule)
        return idx

    def _tensor_flat_index(self, tensor: Tensor, indices: List[int], rule: str, location: SourceLocation) -> int:
        if len(indices) != len(tensor.shape):
            raise ASMRuntimeError("Incorrect number of tensor indices", location=location, rewrite_rule=rule)
        offset = 0
        shape = tensor.shape
        strides = tensor.strides
        for i, raw in enumerate(indices):
            dim_len = shape[i]
            idx = self._resolve_tensor_index(raw, dim_len, rule, location)
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
            if type(item) is TensorLiteral:
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
        while type(current) is IndexExpression:
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
        expression_type = type(expression)
        if expression_type is Literal:
            return Value(expression.literal_type, expression.value)
        if expression_type is TensorLiteral:
            tensor = self._build_tensor_from_literal(expression, env)
            return Value(TYPE_TNS, tensor)
        if expression_type is MapLiteral:
            m = Map()
            eval_expr = self._evaluate_expression
            for key_node, val_node in expression.items:
                key_val = eval_expr(key_node, env)
                key_val = self._deref_value(key_val, location=expression.location, rule="MAP")
                if key_val.type not in (TYPE_INT, TYPE_FLT, TYPE_STR):
                    raise ASMRuntimeError("Map literal keys must be INT, FLT, or STR", location=expression.location, rewrite_rule="MAP")
                key = (key_val.type, key_val.value)
                val = eval_expr(val_node, env)
                m.data[key] = val
            return Value(TYPE_MAP, m)
        if expression_type is PointerExpression:
            return self._make_pointer_value(expression.target, env, expression.location)
        if expression_type is IndexExpression:
            base_expr, index_nodes = self._gather_index_chain(expression)
            base_val = self._evaluate_expression(base_expr, env)
            base_val = self._deref_value(base_val, location=expression.location, rule="INDEX")
            # Tensor indexing path (existing semantics)
            if base_val.type == TYPE_TNS:
                assert isinstance(base_val.value, Tensor)
                eval_expr = self._evaluate_expression
                expect_int = self._expect_int
                # Determine if any index is a range (slice) or star. If none,
                # behave as before and return a single element. If any ranges
                # or stars present, construct numpy slicing indices and return
                # a Tensor.
                RangeT = Range
                StarT = Star
                has_range = any((type(n) is RangeT) or (type(n) is StarT) for n in index_nodes)
                if not has_range:
                    indices: List[int] = []
                    indices_append = indices.append
                    for idx_node in index_nodes:
                        indices_append(expect_int(eval_expr(idx_node, env), "INDEX", expression.location))
                    offset = self._tensor_flat_index(base_val.value, indices, "INDEX", expression.location)
                    return base_val.value.data[offset]

                # Build numpy indexers (mix of ints and slices). Resolve 1-based
                # indices into 0-based python indices; ranges are inclusive.
                arr = base_val.value.data.reshape(tuple(base_val.value.shape))
                indexers: List[object] = []
                for dim, node in enumerate(index_nodes):
                    node_type = type(node)
                    if node_type is Range:
                        lo_val = expect_int(eval_expr(node.lo, env), "INDEX", expression.location)
                        hi_val = expect_int(eval_expr(node.hi, env), "INDEX", expression.location)
                        # validate both endpoints
                        lo_res = self._resolve_tensor_index(lo_val, base_val.value.shape[dim], "INDEX", expression.location)
                        hi_res = self._resolve_tensor_index(hi_val, base_val.value.shape[dim], "INDEX", expression.location)
                        # convert to 0-based slice: start = lo_res-1, stop = hi_res (exclusive)
                        indexers.append(slice(lo_res - 1, hi_res))
                    elif node_type is Star:
                        # Full-dimension slice `*` selects the entire axis
                        indexers.append(slice(None, None))
                    else:
                        raw = expect_int(eval_expr(node, env), "INDEX", expression.location)
                        idx_res = self._resolve_tensor_index(raw, base_val.value.shape[dim], "INDEX", expression.location)
                        indexers.append(idx_res - 1)

                sel = arr[tuple(indexers)]
                sel_arr = sel if isinstance(sel, np.ndarray) else np.array(sel, dtype=object)
                # Ensure sel_arr is at least 1-D (slices guarantee at least one dim)
                if sel_arr.ndim == 0:
                    # Single element selected despite slice usage; wrap as 1-element tensor
                    sel_arr = sel_arr.reshape((1,))
                out_shape = list(sel_arr.shape)
                out_data = sel_arr.ravel()
                return Value(TYPE_TNS, Tensor(shape=out_shape, data=out_data))

            # Map indexing path: support multi-key lookup like m<k1,k2>
            if base_val.type == TYPE_MAP:
                assert isinstance(base_val.value, Map)
                eval_expr = self._evaluate_expression
                current = base_val
                for idx, node in enumerate(index_nodes):
                    key_val = eval_expr(node, env)
                    key_val = self._deref_value(key_val, location=expression.location, rule="INDEX")
                    if key_val.type not in (TYPE_INT, TYPE_FLT, TYPE_STR):
                        raise ASMRuntimeError("Map keys must be INT, FLT, or STR", location=expression.location, rewrite_rule="INDEX")
                    key = (key_val.type, key_val.value)
                    if not isinstance(current.value, Map):
                        raise ASMRuntimeError("Attempted map-indexing into non-map value", location=expression.location, rewrite_rule="INDEX")
                    if key not in current.value.data:
                        raise ASMRuntimeError(f"Key not found: {key_val.value}", location=expression.location, rewrite_rule="INDEX")
                    current = current.value.data[key]
                    if idx + 1 < len(index_nodes):
                        current = self._deref_value(current, location=expression.location, rule="INDEX")
                return current

            raise ASMRuntimeError("Indexed access requires a tensor or map", location=expression.location, rewrite_rule="INDEX")
        if expression_type is Identifier:
            found = env.get_optional(expression.name)
            if found is not None:
                # Normal identifier access should follow pointer aliases.
                # `@name` is the explicit pointer form and returns a PointerRef;
                # a plain `name` should yield the underlying value.
                return self._deref_value(found, location=expression.location, rule="IDENT")
            resolved_name: Optional[str] = None
            if self.call_stack:
                resolved_name = self._resolve_user_function_name(frame_name=self.call_stack[-1].name, callee=expression.name)
            elif expression.name in self.functions:
                resolved_name = expression.name
            if resolved_name is not None:
                return Value(TYPE_FUNC, self.functions[resolved_name])
            if expression.name == "INPUT":
                self._emit_event("before_call", self, "INPUT", [], env, expression.location)
                builtin = self.builtins.table.get("INPUT")
                if builtin is None:
                    raise ASMRuntimeError("Unknown function 'INPUT'", location=expression.location)
                supplied = 0
                if supplied < builtin.min_args:
                    raise ASMRuntimeError(f"INPUT expects at least {builtin.min_args} arguments", rewrite_rule="INPUT", location=expression.location)
                if builtin.max_args is not None and supplied > builtin.max_args:
                    raise ASMRuntimeError(f"INPUT expects at most {builtin.max_args} arguments", rewrite_rule="INPUT", location=expression.location)
                result = builtin.impl(self, [], [], env, expression.location)
                # Start any deferred ASYNC workers now that the builtin had
                # an opportunity to operate on the returned THR (e.g. PAUSE).
                self._flush_pending_async_starts()
                self._emit_event("after_call", self, "INPUT", result, env, expression.location)
                self._log_step(rule="INPUT", location=expression.location, extra={"args": [], "result": result.value})
                return result
            raise ASMRuntimeError(
                f"Undefined identifier '{expression.name}'",
                location=expression.location,
                rewrite_rule="IDENT",
            )
        if expression_type is LambdaExpression:
            # Create an anonymous function value that closes over the current environment.
            self._lambda_counter += 1
            name = f"<lambda_{self._lambda_counter}>"
            fn = Function(
                name=name,
                params=expression.params,
                return_type=expression.return_type,
                body=expression.body,
                closure=env,
            )
            self._log_step(
                rule="LAMBDA",
                location=expression.location,
                extra={
                    "name": name,
                    "params": [p.name for p in expression.params],
                    "return_type": expression.return_type,
                },
            )
            return Value(TYPE_FUNC, fn)
        if expression_type.__name__ == "AsyncExpression":
            # Evaluate ASYNC as an expression: create a THR controller and return it.
            block = expression.block  # type: ignore[attr-defined]
            loc = expression.location

            ctrl = {
                "thread": None,
                "paused": False,
                "pause_event": threading.Event(),
                "finished": False,
                "stop": False,
                "state": "running",
                "env": env,
                "block": block,
            }
            # controller starts unpaused
            ctrl["pause_event"].set()

            def _async_worker_expr():
                frame = self._new_frame("<async>", env, loc)
                self._thread_ctrls[threading.get_ident()] = ctrl
                self.call_stack.append(frame)
                try:
                    self._emit_event("async_start", self, frame, env)
                    try:
                        self._execute_block(block.statements, env)
                    except Exception as exc:
                        try:
                            self._emit_event("on_error", self, exc)
                        except Exception:
                            pass
                finally:
                    try:
                        self.call_stack.pop()
                    except Exception:
                        pass
                    try:
                        tid = threading.get_ident()
                        if tid in self._thread_ctrls:
                            del self._thread_ctrls[tid]
                    except Exception:
                        pass
                    try:
                        self._emit_event("async_end", self, frame, env)
                    except Exception:
                        pass
                    ctrl["finished"] = True
                    ctrl["state"] = "finished" if not ctrl.get("stop") else "stopped"

            t = threading.Thread(target=_async_worker_expr, daemon=True, name=f"asm_async_{self.frame_counter}")
            ctrl["thread"] = t
            # Defer starting the worker thread until the enclosing call
            # completes. This avoids races where a builtin (e.g. PAUSE)
            # receives the THR and attempts to pause it only after the
            # worker has already begun executing.
            ctrl["start_pending"] = True
            self._pending_async_starts.append(ctrl)
            return Value(TYPE_THR, ctrl)
        if expression_type is CallExpression:
            callee_expr = expression.callee
            callee_ident = callee_expr.name if type(callee_expr) is Identifier else None

            bound_func_value = env.get_optional(callee_ident) if callee_ident is not None else None
            resolved_function: Optional[Function] = None
            alias_name: Optional[str] = None

            if bound_func_value is not None:
                resolved_function = self._expect_func(bound_func_value, "CALL", expression.location)
                alias_name = callee_ident
            elif callee_ident is not None:
                func_name: Optional[str] = None
                if self.call_stack:
                    func_name = self._resolve_user_function_name(frame_name=self.call_stack[-1].name, callee=callee_ident)
                elif callee_ident in self.functions:
                    func_name = callee_ident
                if func_name is not None:
                    resolved_function = self.functions[func_name]
                    alias_name = callee_ident

            if resolved_function is None and callee_ident == "IMPORT":
                if any(arg.name for arg in expression.args):
                    raise ASMRuntimeError("IMPORT does not accept keyword arguments", location=expression.location, rewrite_rule="IMPORT")
                first_expr = expression.args[0].expression if expression.args else None
                module_label = first_expr.name if (first_expr is not None and type(first_expr) is Identifier) else None
                dummy_args: List[Value] = [Value(TYPE_INT, 0)] * len(expression.args)
                arg_nodes = [arg.expression for arg in expression.args]
                try:
                    self._emit_event("before_call", self, "IMPORT", [], env, expression.location)
                    builtin = self.builtins.table.get("IMPORT")
                    if builtin is None:
                        raise ASMRuntimeError("Unknown function 'IMPORT'", location=expression.location)
                    supplied = len(dummy_args)
                    if supplied < builtin.min_args:
                        raise ASMRuntimeError(f"IMPORT expects at least {builtin.min_args} arguments", rewrite_rule="IMPORT", location=expression.location)
                    if builtin.max_args is not None and supplied > builtin.max_args:
                        raise ASMRuntimeError(f"IMPORT expects at most {builtin.max_args} arguments", rewrite_rule="IMPORT", location=expression.location)
                    result = builtin.impl(self, dummy_args, arg_nodes, env, expression.location)
                    self._flush_pending_async_starts()
                except ASMRuntimeError:
                    self._log_step(rule="IMPORT", location=expression.location, extra={"module": module_label, "status": "error"})
                    raise
                self._emit_event("after_call", self, "IMPORT", result, env, expression.location)
                self._log_step(rule="IMPORT", location=expression.location, extra={"module": module_label, "result": result.value})
                return result

            if resolved_function is None and callee_ident in ("DEL", "EXIST"):
                if any(arg.name for arg in expression.args):
                    raise ASMRuntimeError(
                        f"{callee_ident} does not accept keyword arguments",
                        location=expression.location,
                        rewrite_rule=callee_ident,
                    )
                dummy_args = [Value(TYPE_INT, 0)] * len(expression.args)
                arg_nodes = [arg.expression for arg in expression.args]
                try:
                    self._emit_event("before_call", self, callee_ident, [], env, expression.location)
                    builtin = self.builtins.table.get(callee_ident)
                    if builtin is None:
                        raise ASMRuntimeError(f"Unknown function '{callee_ident}'", location=expression.location)
                    supplied = len(dummy_args)
                    if supplied < builtin.min_args:
                        raise ASMRuntimeError(f"{callee_ident} expects at least {builtin.min_args} arguments", rewrite_rule=callee_ident, location=expression.location)
                    if builtin.max_args is not None and supplied > builtin.max_args:
                        raise ASMRuntimeError(f"{callee_ident} expects at most {builtin.max_args} arguments", rewrite_rule=callee_ident, location=expression.location)
                    result = builtin.impl(self, dummy_args, arg_nodes, env, expression.location)
                    self._flush_pending_async_starts()
                except ASMRuntimeError:
                    self._log_step(rule=callee_ident, location=expression.location, extra={"args": None, "status": "error"})
                    raise
                self._emit_event("after_call", self, callee_ident, result, env, expression.location)
                self._log_step(rule=callee_ident, location=expression.location, extra={"args": None, "result": result.value})
                return result

            positional_args: List[Value] = []
            keyword_args: Dict[str, Value] = {}
            eval_expr = self._evaluate_expression
            pointer_args: List[PointerRef] = []
            for arg in expression.args:
                value = eval_expr(arg.expression, env)
                if isinstance(value.value, PointerRef):
                    pointer_args.append(value.value)
                if arg.name is None:
                    positional_args.append(value)
                else:
                    if arg.name in keyword_args:
                        raise ASMRuntimeError(
                            f"Duplicate keyword argument '{arg.name}'",
                            location=expression.location,
                            rewrite_rule=callee_ident or "CALL",
                        )
                    keyword_args[arg.name] = value

            if resolved_function is not None:
                return self._invoke_function_object(resolved_function, positional_args, keyword_args, expression.location, env, alias=alias_name)

            if callee_ident is not None:
                try:
                    if keyword_args:
                        if callee_ident in {"READFILE", "WRITEFILE"}:
                            allowed = {"coding"}
                            key = "coding"
                        elif callee_ident == "BYTES":
                            allowed = {"endian"}
                            key = "endian"
                        else:
                            allowed = set()
                            key = None

                        unexpected = [k for k in keyword_args if k not in allowed]
                        if unexpected:
                            raise ASMRuntimeError(
                                f"Unexpected keyword arguments: {', '.join(sorted(unexpected))}",
                                location=expression.location,
                                rewrite_rule=callee_ident,
                            )
                        if key and key in keyword_args:
                            positional_args.append(keyword_args.pop(key))
                        if keyword_args:
                            raise ASMRuntimeError(
                                f"{callee_ident} does not accept keyword arguments",
                                location=expression.location,
                                rewrite_rule=callee_ident,
                            )
                    arg_nodes = [a.expression for a in expression.args]
                    self._emit_event("before_call", self, callee_ident, positional_args, env, expression.location)
                    builtin = self.builtins.table.get(callee_ident)
                    if builtin is None:
                        raise ASMRuntimeError(f"Unknown function '{callee_ident}'", location=expression.location)
                    supplied = len(positional_args)
                    if supplied < builtin.min_args:
                        raise ASMRuntimeError(f"{callee_ident} expects at least {builtin.min_args} arguments", rewrite_rule=callee_ident, location=expression.location)
                    if builtin.max_args is not None and supplied > builtin.max_args:
                        raise ASMRuntimeError(f"{callee_ident} expects at most {builtin.max_args} arguments", rewrite_rule=callee_ident, location=expression.location)
                    result = builtin.impl(self, positional_args, arg_nodes, env, expression.location)
                    self._flush_pending_async_starts()
                    if pointer_args:
                        self._assign_pointer_target(pointer_args[0], result, location=expression.location, rule=callee_ident)
                except ASMRuntimeError:
                    self._log_step(
                        rule=callee_ident,
                        location=expression.location,
                        extra={
                            "args": [a.value for a in positional_args],
                            "keyword": {k: v.value for k, v in keyword_args.items()},
                            "status": "error",
                        },
                    )
                    raise
                self._emit_event("after_call", self, callee_ident, result, env, expression.location)
                self._log_step(
                    rule=callee_ident,
                    location=expression.location,
                    extra={
                        "args": [a.value for a in positional_args],
                        "keyword": {k: v.value for k, v in keyword_args.items()},
                        "result": result.value,
                    },
                )
                return result

            callee_value = self._evaluate_expression(callee_expr, env)
            func_obj = self._expect_func(callee_value, "CALL", expression.location)
            return self._invoke_function_object(func_obj, positional_args, keyword_args, expression.location, env)
        raise ASMRuntimeError("Unsupported expression", location=expression.location)

    def _invoke_function_object(
        self,
        func: Function,
        positional_args: List[Value],
        keyword_args: Dict[str, Value],
        call_location: SourceLocation,
        env: Environment,
        *,
        alias: Optional[str] = None,
    ) -> Value:
        label = alias or func.name
        self._log_step(
            rule="CALL",
            location=call_location,
            extra={
                "function": label,
                "target": func.name,
                "positional": [a.value for a in positional_args],
                "keyword": {k: v.value for k, v in keyword_args.items()},
            },
        )
        self._emit_event("before_call", self, func.name, positional_args, env, call_location)
        return self._call_user_function(func, positional_args, keyword_args, call_location)

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
            if function.return_type == TYPE_FUNC:
                raise ASMRuntimeError(
                    f"Function {function.name} must return a function value",
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
        # Hot-path: avoid a function call + loop when no handlers exist.
        handlers = self.hook_registry._events.get(event)
        if not handlers:
            return
        try:
            for _priority, handler, _ext in handlers:
                handler(*args, **kwargs)
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

    def _flush_pending_async_starts(self) -> None:
        """Start any ASYNC workers that were deferred during expression evaluation.

        The list is cleared after starting so each pending controller is only
        started once. Exceptions while starting a thread are swallowed to
        avoid turning benign thread-start failures into interpreter crashes.
        """
        if not getattr(self, "_pending_async_starts", None):
            return
        pending = self._pending_async_starts
        self._pending_async_starts = []
        for ctrl in pending:
            if not ctrl.get("start_pending"):
                continue
            try:
                ctrl["thread"].start()
            except Exception:
                pass
            ctrl["start_pending"] = False

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
        # Hot-path: skip context allocation + dispatch if none are registered.
        if not self.hook_registry._step_rules:
            return
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
