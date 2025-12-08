from __future__ import annotations
import json
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union

from lexer import ASMError, ASMParseError, Lexer
from parser import (
    Assignment,
    Block,
    BreakStatement,
    CallExpression,
    Expression,
    ExpressionStatement,
    ForStatement,
    FuncDef,
    GotoStatement,
    GotopointStatement,
    Identifier,
    IfBranch,
    IfStatement,
    Literal,
    Parser,
    Program,
    ReturnStatement,
    SourceLocation,
    Statement,
    WhileStatement,
    ContinueStatement,
)


TYPE_INT = "INT"
TYPE_STR = "STR"


@dataclass
class Value:
    type: str
    value: Union[int, str]


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
            del env.values[name]
            return
        raise ASMRuntimeError(f"Cannot delete undefined identifier '{name}'", rewrite_rule="DEL")

    def has(self, name: str) -> bool:
        return self._find_env(name) is not None

    def snapshot(self) -> Dict[str, str]:
        return {k: f"{v.type}:{v.value}" for k, v in self.values.items()}


@dataclass
class Function:
    name: str
    params: List[str]
    param_types: List[str]
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
        self._register_custom("EQ", 2, 2, self._eq)
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
        self._register_variadic("JOIN", 1, self._join)
        self._register_int_only("LOG", 1, self._safe_log)
        self._register_int_only("CLOG", 1, self._safe_clog)
        self._register_custom("INT", 1, 1, self._int_op)
        self._register_custom("STR", 1, 1, self._str_op)
        self._register_custom("MAIN", 0, 0, self._main)
        self._register_custom("IMPORT", 1, 1, self._import)
        self._register_custom("INPUT", 0, 0, self._input)
        self._register_custom("PRINT", 0, None, self._print)
        self._register_custom("ASSERT", 1, 1, self._assert)
        self._register_custom("DEL", 1, 1, self._delete)
        self._register_custom("EXIST", 1, 1, self._exist)
        self._register_custom("EXIT", 0, 1, self._exit)

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

    def _as_bool_value(self, value: Value) -> int:
        if value.type == TYPE_INT:
            return 0 if value.value == 0 else 1
        if value.type == TYPE_STR:
            return 0 if value.value == "" else 1
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
        return Value(TYPE_INT, len(values))

    def _join(self, values: List[Value], location: SourceLocation) -> Value:
        if not values:
            raise ASMRuntimeError("JOIN requires at least one argument", rewrite_rule="JOIN")
        first_type = values[0].type
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

    def _eq(self, _: "Interpreter", args: List[Value], __: List[Expression], ___: Environment, ___loc: SourceLocation) -> Value:
        a, b = args
        if a.type != b.type:
            return Value(TYPE_INT, 0)
        return Value(TYPE_INT, 1 if a.value == b.value else 0)

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

        new_funcs = {n: f for n, f in interpreter.functions.items() if n not in prev_functions}
        for name, fn in new_funcs.items():
            dotted_name = f"{module_name}.{name}"
            if "." in name:
                interpreter.functions[dotted_name] = Function(
                    name=dotted_name,
                    params=fn.params,
                    param_types=fn.param_types,
                    return_type=fn.return_type,
                    body=fn.body,
                    closure=fn.closure,
                )
            else:
                interpreter.functions.pop(name, None)
                interpreter.functions[dotted_name] = Function(
                    name=dotted_name,
                    params=fn.params,
                    param_types=fn.param_types,
                    return_type=fn.return_type,
                    body=fn.body,
                    closure=module_env,
                )

        for k, v in module_env.values.items():
            dotted = f"{module_name}.{k}"
            env.set(dotted, v, declared_type=v.type)
        return Value(TYPE_INT, 0)

    def _input(
        self,
        interpreter: "Interpreter",
        args: List[Value],
        __: List[Expression],
        ___: Environment,
        ____: SourceLocation,
    ) -> Value:
        text = interpreter.input_provider()
        interpreter.io_log.append({"event": "INPUT", "text": text})
        return Value(TYPE_STR, text)

    def _print(
        self,
        interpreter: "Interpreter",
        args: List[Value],
        __: List[Expression],
        ___: Environment,
        ____: SourceLocation,
    ) -> Value:
        rendered: List[str] = []
        for arg in args:
            if arg.type == TYPE_INT:
                number = self._expect_int(arg, "PRINT", ____)
                rendered.append(("-" + format(-number, "b")) if number < 0 else format(number, "b"))
            elif arg.type == TYPE_STR:
                rendered.append(arg.value)  # type: ignore[arg-type]
            else:
                rendered.append(str(arg.value))
        text = " ".join(rendered)
        interpreter.output_sink(text)
        interpreter.io_log.append({"event": "PRINT", "values": [arg.value for arg in args]})
        return Value(TYPE_INT, 0)

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
            param_types = [ptype for ptype, _ in statement.params]
            param_names = [pname for _, pname in statement.params]
            self.functions[statement.name] = Function(
                name=statement.name,
                params=param_names,
                param_types=param_types,
                return_type=statement.return_type,
                body=statement.body,
                closure=env,
            )
            return
        if isinstance(statement, ReturnStatement):
            frame: Frame = self.call_stack[-1]
            if frame.name == "<top-level>":
                raise ASMRuntimeError("RETURN outside of function", location=statement.location, rewrite_rule="RETURN")
            value = self._evaluate_expression(statement.expression, env) if statement.expression else Value(TYPE_INT, 0)
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
        raise ASMRuntimeError("Unsupported type in condition", location=location)

    def _expect_int(self, value: Value, rule: str, location: Optional[SourceLocation]) -> int:
        if value.type != TYPE_INT:
            raise ASMRuntimeError(f"{rule} expects integer value", location=location, rewrite_rule=rule)
        return value.value  # type: ignore[return-value]

    def _evaluate_expression(self, expression: Expression, env: Environment) -> Value:
        if isinstance(expression, Literal):
            return Value(expression.literal_type, expression.value)
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
                module_label = expression.args[0].name if (expression.args and isinstance(expression.args[0], Identifier)) else None
                dummy_args: List[Value] = [Value(TYPE_INT, 0)] * len(expression.args)
                try:
                    result = self.builtins.invoke(self, expression.name, dummy_args, expression.args, env, expression.location)
                except ASMRuntimeError:
                    self._log_step(rule="IMPORT", location=expression.location, extra={"module": module_label, "status": "error"})
                    raise
                self._log_step(rule="IMPORT", location=expression.location, extra={"module": module_label, "result": result.value})
                return result
            if expression.name in ("DEL", "EXIST"):
                dummy_args: List[Value] = [Value(TYPE_INT, 0)] * len(expression.args)
                try:
                    result = self.builtins.invoke(self, expression.name, dummy_args, expression.args, env, expression.location)
                except ASMRuntimeError:
                    self._log_step(rule=expression.name, location=expression.location, extra={"args": None, "status": "error"})
                    raise
                self._log_step(rule=expression.name, location=expression.location, extra={"args": None, "result": result.value})
                return result
            args: List[Value] = []
            for arg in expression.args:
                args.append(self._evaluate_expression(arg, env))
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
                self._log_step(rule="CALL", location=expression.location, extra={"function": func_name, "args": [a.value for a in args]})
                return self._call_user_function(self.functions[func_name], args, expression.location)
            try:
                result = self.builtins.invoke(self, expression.name, args, expression.args, env, expression.location)
            except ASMRuntimeError:
                self._log_step(rule=expression.name, location=expression.location, extra={"args": [a.value for a in args], "status": "error"})
                raise
            self._log_step(rule=expression.name, location=expression.location, extra={"args": [a.value for a in args], "result": result.value})
            return result
        raise ASMRuntimeError("Unsupported expression", location=expression.location)

    def _call_user_function(self, function: Function, args: List[Value], call_location: SourceLocation) -> Value:
        if len(args) != len(function.params):
            raise ASMRuntimeError(
                f"Function {function.name} expects {len(function.params)} arguments but received {len(args)}",
                location=call_location,
            )
        env = Environment(parent=function.closure)
        for param_name, param_type, arg in zip(function.params, function.param_types, args):
            if arg.type != param_type:
                raise ASMRuntimeError(
                    f"Argument for '{param_name}' expected {param_type} but got {arg.type}",
                    location=call_location,
                    rewrite_rule=function.name,
                )
            env.set(param_name, arg, declared_type=param_type)
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
            return Value(TYPE_STR, "")

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
