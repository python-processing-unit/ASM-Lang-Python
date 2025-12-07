"""Reference interpreter for ASM-Lang (spec.txt)."""
from __future__ import annotations
import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional
class ASMError(Exception):
    """Base class for interpreter errors."""
class ASMParseError(ASMError):
    """Raised when parsing fails."""
class ASMRuntimeError(ASMError):
    """Raised for runtime faults."""
    def __init__(
        self,
        message: str,
        *,
        location: Optional["SourceLocation"] = None,
        rewrite_rule: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.location = location
        self.rewrite_rule = rewrite_rule
        self.step_index: Optional[int] = None
class ReturnSignal(Exception):
    def __init__(self, value: int) -> None:
        super().__init__(value)
        self.value = value
class ExitSignal(Exception):
    def __init__(self, code: int = 0) -> None:
        super().__init__(code)
        self.code = code
@dataclass
class SourceLocation:
    file: str
    line: int
    column: int
    statement: str
@dataclass
class Token:
    type: str
    value: str
    line: int
    column: int
KEYWORDS = {"IF", "ELSIF", "ELSE", "WHILE", "FOR", "FUNC", "RETURN"}
SYMBOLS = {
    "(": "LPAREN",
    ")": "RPAREN",
    "[": "LBRACKET",
    "]": "RBRACKET",
    "{": "LBRACE",
    "}": "RBRACE",
    ",": "COMMA",
    "=": "EQUALS",
}
class Lexer:
    def __init__(self, text: str, filename: str) -> None:
        self.text = text
        self.filename = filename
        self.index = 0
        self.line = 1
        self.column = 1
    def tokenize(self) -> List[Token]:
        tokens: List[Token] = []
        # Cache frequently used methods/values locally for speed.
        _peek = self._peek
        _advance = self._advance
        _is_identifier_start = self._is_identifier_start
        symbols = SYMBOLS
        while not self._eof:
            ch: str = _peek()
            if(ch in " \t"):
                _advance()
                continue
            if(ch == "\r"):
                _advance()
                continue
            if(ch == "\n"):
                tokens.append(Token("NEWLINE","\n",self.line,self.column))
                _advance()
                continue
            if(ch == "#"):
                self._consume_comment()
                continue
            if(ch in SYMBOLS):
                tokens.append(Token(symbols[ch], ch, self.line, self.column))
                _advance()
                continue
            if(ch == "-"):
                tokens.append(self._consume_signed_number())
                continue
            if(ch in "01"):
                tokens.append(self._consume_unsigned_number())
                continue
            if _is_identifier_start(ch):
                tokens.append(self._consume_identifier())
                continue
            raise ASMParseError(f"Unexpected character '{ch}' at {self.filename}:{self.line}:{self.column}")
        tokens.append(Token("EOF","",self.line,self.column))
        return(tokens)
    def _consume_comment(self) -> None:
        while(not self._eof and self._peek() != "\n"):
            self._advance()
    def _consume_unsigned_number(self) -> Token:
        line, col = self.line, self.column
        digits = self._consume_binary_digits()
        return(Token("NUMBER",digits,line,col))

    def _consume_signed_number(self) -> Token:
        line, col = self.line, self.column
        self._advance()  # consume '-'
        while(not self._eof and self._peek() in " \t\r"):
            self._advance()
        if(self._eof or self._peek() not in "01"):
            raise ASMParseError(f"Expected binary digits after '-' at {self.filename}:{line}:{col}")
        digits = self._consume_binary_digits()
        return(Token("NUMBER","-" + digits,line,col))

    def _consume_binary_digits(self) -> str:
        digits: List[str] = []
        while(not self._eof and self._peek() in "01"):
            digits.append(self._peek())
            self._advance()
        return "".join(digits)
    def _consume_identifier(self) -> Token:
        line, col = self.line, self.column
        # Defensive check: identifiers must not start with the digits '0' or '1'.
        # Normally this is enforced by _is_identifier_start, but check here
        # so that the lexer produces a clear parse error if invoked incorrectly.
        if not self._eof and self._peek() in "01":
            raise ASMParseError(
                f"Identifiers must not start with '0' or '1' at {self.filename}:{line}:{col}"
            )
        chars: List[str] = []
        while(not self._eof and self._is_identifier_part(self._peek())):
            chars.append(self._peek())
            self._advance()
        value = "".join(chars)
        token_type: str = value if value in KEYWORDS else "IDENT"
        return Token(token_type, value, line, col)
    def _is_identifier_start(self, ch: str) -> bool:
        # Only allow ASCII letters and underscore as the start of an identifier.
        # This enforces the spec requirement that identifiers must be ASCII-only.
        return (ch == "_") or ("A" <= ch <= "Z") or ("a" <= ch <= "z")
    def _is_identifier_part(self, ch: str) -> bool:
        # Allow ASCII letters, underscore, and digits (including '0' and '1').
        # The first character of an identifier must not be '0' or '1'
        # (this is enforced by _is_identifier_start and by a defensive
        # check in _consume_identifier). Characters '0' and '1' are permitted
        # in subsequent positions inside an identifier.
        return (
            (ch == "_")
            or ("A" <= ch <= "Z")
            or ("a" <= ch <= "z")
            or ("0" <= ch <= "9")
        )
    @property
    def _eof(self) -> bool:
        return(self.index >= len(self.text))
    def _peek(self) -> str:
        return(self.text[self.index])
    def _advance(self) -> None:
        if(self.text[self.index] == "\n"):
            self.line += 1
            self.column = 1
        else:
            self.column += 1
        self.index += 1
@dataclass
class Node:
    location: SourceLocation
@dataclass
class Program(Node):
    statements: List["Statement"]
class Statement(Node):
    pass
@dataclass
class Block(Node):
    statements: List[Statement]
@dataclass
class Assignment(Statement):
    target: str
    expression: "Expression"
@dataclass
class ExpressionStatement(Statement):
    expression: "Expression"
@dataclass
class IfBranch:
    condition: "Expression"
    block: Block
@dataclass
class IfStatement(Statement):
    condition: "Expression"
    then_block: Block
    elifs: List[IfBranch]
    else_block: Optional[Block]
@dataclass
class WhileStatement(Statement):
    condition: "Expression"
    block: Block
@dataclass
class ForStatement(Statement):
    counter: str
    target_expr: "Expression"
    block: Block
@dataclass
class FuncDef(Statement):
    name: str
    params: List[str]
    body: Block
@dataclass
class ReturnStatement(Statement):
    expression: Optional["Expression"]
class Expression(Node):
    pass
@dataclass
class Literal(Expression):
    value: int
@dataclass
class Identifier(Expression):
    name: str
@dataclass
class CallExpression(Expression):
    name: str
    args: List[Expression]
class Parser:
    def __init__(self, tokens: List[Token], filename: str, source_lines: List[str]):
        self.tokens = tokens
        self.filename = filename
        self.source_lines = source_lines
        self.index = 0
    def parse(self) -> Program:
        statements:List[Statement] = self._parse_statements(stop_tokens={"EOF"})
        eof_token:Token = self._peek()
        return(Program(location=self._location_from_token(eof_token),statements=statements))
    def _parse_statements(self, stop_tokens: Iterable[str]) -> List[Statement]:
        statements: List[Statement] = []
        while(self._peek().type not in stop_tokens):
            if(self._match("NEWLINE")):
                continue
            statements.append(self._parse_statement())
            self._consume_newlines()
        return(statements)
    def _parse_statement(self) -> Statement:
        token = self._peek()
        if(token.type == "FUNC"):
            return(self._parse_func())
        if(token.type == "IF"):
            return(self._parse_if())
        if(token.type == "WHILE"):
            return(self._parse_while())
        if(token.type == "FOR"):
            return(self._parse_for())
        if(token.type == "RETURN"):
            return(self._parse_return())
        if(token.type == "IDENT" and self._peek_next().type == "EQUALS"):
            return(self._parse_assignment())
        expr:Expression = self._parse_expression()
        return(ExpressionStatement(location=expr.location,expression=expr))
    def _parse_assignment(self) -> Assignment:
        ident = self._consume("IDENT")
        self._consume("EQUALS")
        expr = self._parse_expression()
        location:SourceLocation = self._location_from_token(ident)
        return(Assignment(location=location,target=ident.value,expression=expr))
    def _parse_func(self) -> FuncDef:
        keyword = self._consume("FUNC")
        name_token = self._consume("IDENT")
        self._consume("LPAREN")
        params: List[str] = []
        if(self._peek().type != "RPAREN"):
            while True:
                params.append(self._consume("IDENT").value)
                if(not self._match("COMMA")):
                    break
        self._consume("RPAREN")
        block:Block = self._parse_block()
        location:SourceLocation = self._location_from_token(keyword)
        return(FuncDef(location=location,name=name_token.value,params=params,body=block))
    def _parse_if(self) -> IfStatement:
        keyword = self._consume("IF")
        condition:Expression = self._parse_parenthesized_expression()
        then_block:Block = self._parse_block()
        elifs:List[IfBranch] = []
        while(self._match("ELSIF")):
            cond:Expression = self._parse_parenthesized_expression()
            block:Block = self._parse_block()
            elifs.append(IfBranch(condition=cond,block=block))
        else_block:Optional[Block] = self._parse_block() if(self._match("ELSE")) else None
        return(IfStatement(location=self._location_from_token(keyword),condition=condition,then_block=then_block,elifs=elifs,else_block=else_block))
    def _parse_while(self) -> WhileStatement:
        keyword = self._consume("WHILE")
        condition:Expression = self._parse_parenthesized_expression()
        block:Block = self._parse_block()
        return(WhileStatement(location=self._location_from_token(keyword),condition=condition,block=block))
    def _parse_for(self) -> ForStatement:
        keyword = self._consume("FOR")
        self._consume("LPAREN")
        counter = self._consume("IDENT")
        self._consume("COMMA")
        target:Expression = self._parse_expression()
        self._consume("RPAREN")
        block:Block = self._parse_block()
        return(ForStatement(location=self._location_from_token(keyword),counter=counter.value,target_expr=target,block=block))
    def _parse_return(self) -> ReturnStatement:
        keyword = self._consume("RETURN")
        expression:Expression = self._parse_parenthesized_expression()
        return(ReturnStatement(location=self._location_from_token(keyword),expression=expression))
    def _parse_block(self) -> Block:
        opening = self._peek().type
        if(opening == "LBRACKET"):
            start = self._consume("LBRACKET")
            closing = "RBRACKET"
        elif(opening == "LBRACE"):
            start = self._consume("LBRACE")
            closing = "RBRACE"
        else:
            raise ASMParseError(f"Expected '[' or '{{' to start block but found {opening}")
        statements:List[Statement] = self._parse_statements(stop_tokens={closing})
        self._consume(closing)
        return(Block(location=self._location_from_token(start),statements=statements))
    def _parse_expression(self) -> Expression:
        token = self._peek()
        if(token.type == "NUMBER"):
            number:Token = self._consume("NUMBER")
            # NUMBER token may be prefixed with '-' for negative literals.
            sval = number.value
            if sval.startswith("-"):
                # safe parse of negative binary literal
                if len(sval) == 1:
                    raise ASMParseError(f"Invalid numeric literal at line {number.line}")
                value = -int(sval[1:], 2)
            else:
                value = int(sval, 2)
            return Literal(location=self._location_from_token(number), value=value)
        if(token.type == "IDENT"):
            ident:Token = self._consume("IDENT")
            location:SourceLocation = self._location_from_token(ident)
            if(self._match("LPAREN")):
                args:List[Expression] = []
                if(self._peek().type != "RPAREN"):
                    while True:
                        args.append(self._parse_expression())
                        if(not self._match("COMMA")):
                            break
                self._consume("RPAREN")
                return(CallExpression(location=location,name=ident.value,args=args))
            return(Identifier(location=location,name=ident.value))
        if(token.type == "LPAREN"):
            self._consume("LPAREN")
            expr:Expression = self._parse_expression()
            self._consume("RPAREN")
            return(expr)
        raise ASMParseError(f"Unexpected token {token.type} in expression at line {token.line}")
    def _parse_parenthesized_expression(self) -> Expression:
        self._consume("LPAREN")
        expr = self._parse_expression()
        self._consume("RPAREN")
        return expr
    def _consume(self, token_type: str) -> Token:
        token = self._peek()
        if(token.type != token_type):
            raise ASMParseError(f"Expected token {token_type} but found {token.type} at line {token.line}")
        self.index += 1
        return token
    def _match(self, token_type: str) -> bool:
        if(self._peek().type == token_type):
            self.index += 1
            return True
        return False
    def _consume_newlines(self) -> None:
        while(self._match("NEWLINE")):
            continue
    def _peek(self) -> Token:
        return self.tokens[self.index]
    def _peek_next(self) -> Token:
        return self.tokens[self.index + 1]
    def _location_from_token(self, token: Token) -> SourceLocation:
        line_index = token.line - 1
        statement = ""
        if 0 <= line_index < len(self.source_lines):
            statement = self.source_lines[line_index].strip()
        return SourceLocation(file=self.filename,line=token.line,column=token.column,statement=statement)
class Environment:
    def __init__(self, parent: Optional["Environment"] = None) -> None:
        self.parent = parent
        self.values: Dict[str, int] = {}
    def set(self, name: str, value: int) -> None:
        # Assign to the nearest environment that already contains the name.
        # If the name does not exist in any parent, create it in this env.
        env: Optional[Environment] = self
        while env is not None:
            if name in env.values:
                env.values[name] = value
                return
            env = env.parent
        # Not found in parents: bind in the current environment
        self.values[name] = value
    def get(self, name: str) -> int:
        if name in self.values:
            return self.values[name]
        if self.parent is not None:
            return self.parent.get(name)
        raise ASMRuntimeError(f"Undefined identifier '{name}'", rewrite_rule="IDENT")

    def delete(self, name: str) -> None:
        # Delete the identifier from the nearest environment that defines it.
        env: Optional[Environment] = self
        while env is not None:
            if name in env.values:
                del env.values[name]
                return
            env = env.parent
        raise ASMRuntimeError(f"Cannot delete undefined identifier '{name}'", rewrite_rule="DEL")
    def has(self, name: str) -> bool:
        """Return True if 'name' exists in this environment or any parent."""
        if name in self.values:
            return True
        if self.parent is not None:
            return self.parent.has(name)
        return False
    def snapshot(self) -> Dict[str, int]:
        return dict(self.values)
@dataclass
class Function:
    name: str
    params: List[str]
    body: Block
    closure: Environment
@dataclass
class Frame:
    name: str
    env: Environment
    frame_id: str
    call_location: Optional[SourceLocation]
@dataclass
class StateEntry:
    step_index: int
    state_id: str
    frame_id: Optional[str]
    source_location: Optional[SourceLocation]
    statement: Optional[str]
    env_snapshot: Optional[Dict[str, int]]
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
BuiltinImpl = Callable[["Interpreter", List[int], List[Expression], Environment, SourceLocation], int]
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
        self._register_fixed("ADD", 2, lambda a, b: a + b)
        self._register_fixed("SUB", 2, lambda a, b: a - b)
        self._register_fixed("MUL", 2, lambda a, b: a * b)
        self._register_fixed("DIV", 2, self._safe_div)
        self._register_fixed("CDIV", 2, self._safe_cdiv)
        self._register_fixed("MOD", 2, self._safe_mod)
        self._register_fixed("POW", 2, self._safe_pow)
        self._register_fixed("NEG", 1, lambda a: -a)
        self._register_fixed("ABS", 1, abs)
        self._register_fixed("GCD", 2, math.gcd)
        self._register_fixed("LCM", 2, self._lcm)
        self._register_fixed("BAND", 2, lambda a, b: a & b)
        self._register_fixed("BOR", 2, lambda a, b: a | b)
        self._register_fixed("BXOR", 2, lambda a, b: a ^ b)
        self._register_fixed("BNOT", 1, lambda a: ~a)
        self._register_fixed("SHL", 2, self._shift_left)
        self._register_fixed("SHR", 2, self._shift_right)
        # Return the inclusive bit-slice [hi:lo] of `a` as an unsigned integer.
        # Bits are numbered starting at 0 for the least-significant bit.
        # Implementation note: uses masking so negative values yield their
        # two's-complement low bits (consistent with other bitwise ops).
        self._register_fixed("SLICE", 3, self._slice)
        self._register_fixed("AND", 2, lambda a, b: 1 if (_as_bool(a) and _as_bool(b)) else 0)
        self._register_fixed("OR", 2, lambda a, b: 1 if (_as_bool(a) or _as_bool(b)) else 0)
        self._register_fixed("XOR", 2, lambda a, b: 1 if (_as_bool(a) ^ _as_bool(b)) else 0)
        self._register_fixed("NOT", 1, lambda a: 1 if a == 0 else 0)
        self._register_fixed("EQ", 2, lambda a, b: 1 if a == b else 0)
        self._register_fixed("GT", 2, lambda a, b: 1 if a > b else 0)
        self._register_fixed("LT", 2, lambda a, b: 1 if a < b else 0)
        self._register_fixed("GTE", 2, lambda a, b: 1 if a >= b else 0)
        self._register_fixed("LTE", 2, lambda a, b: 1 if a <= b else 0)
        self._register_variadic("SUM", 1, sum)
        self._register_variadic("PROD", 1, self._prod)
        self._register_variadic("MAX", 1, max)
        self._register_variadic("MIN", 1, min)
        self._register_variadic("ANY", 1, lambda vals: 1 if any(_as_bool(v) for v in vals) else 0)
        self._register_variadic("ALL", 1, lambda vals: 1 if all(_as_bool(v) for v in vals) else 0)
        self._register_variadic("LEN", 0, lambda vals: len(vals))
        self._register_variadic("JOIN", 1, self._join)
        self._register_fixed("LOG", 1, self._safe_log)
        self._register_fixed("CLOG", 1, self._safe_clog)
        self._register_custom("MAIN", 0, 0, self._main)
        self._register_custom("IMPORT", 1, 1, self._import)
        self._register_custom("INPUT", 0, 0, self._input)
        self._register_custom("PRINT", 0, None, self._print)
        self._register_custom("ASSERT", 1, 1, self._assert)
        self._register_custom("DEL", 1, 1, self._delete)
        self._register_custom("EXIST", 1, 1, self._exist)
        self._register_custom("EXIT", 0, 1, self._exit)
    def _register_fixed(self, name: str, arity: int, func: Callable[..., int]) -> None:
        def impl(interpreter: "Interpreter", args: List[int], _: List[Expression], __: Environment, ___: SourceLocation) -> int:
            return func(*args)
        self.table[name] = BuiltinFunction(name=name, min_args=arity, max_args=arity, impl=impl)
    def _register_variadic(self, name: str, min_args: int, func: Callable[[List[int]], int]) -> None:
        def impl(interpreter: "Interpreter", args: List[int], _: List[Expression], __: Environment, ___: SourceLocation) -> int:
            return func(args)
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
        args: List[int],
        arg_nodes: List[Expression],
        env: Environment,
        location: SourceLocation,
    ) -> int:
        builtin = self.table.get(name)
        if builtin is None:
            raise ASMRuntimeError(f"Unknown function '{name}'", location=location)
        # Inline validation to avoid a method call in hot path.
        supplied = len(args)
        if supplied < builtin.min_args:
            raise ASMRuntimeError(f"{name} expects at least {builtin.min_args} arguments", rewrite_rule=name)
        if builtin.max_args is not None and supplied > builtin.max_args:
            raise ASMRuntimeError(f"{name} expects at most {builtin.max_args} arguments", rewrite_rule=name)
        return builtin.impl(interpreter, args, arg_nodes, env, location)
    def _safe_div(self, a: int, b: int) -> int:
        if b == 0:
            raise ASMRuntimeError("Division by zero", rewrite_rule="DIV")
        return a // b
    def _safe_cdiv(self, a: int, b: int) -> int:
        if b == 0:
            raise ASMRuntimeError("Division by zero", rewrite_rule="CDIV")
        # Compute ceiling division CDIV(a / b) using integer operations only.
        # Use Python's floor division to get the floor quotient `q`.
        q = a // b
        # If remainder is zero then the division is exact; otherwise ceil is q+1.
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
    def _prod(self, values: List[int]) -> int:
        result = 1
        for value in values:
            result *= value
        return result
    def _join(self, values: List[int]) -> int:
        # Allow either all non-negative arguments or all negative arguments.
        if any(value < 0 for value in values):
            if not all(value < 0 for value in values):
                raise ASMRuntimeError("JOIN arguments must not mix positive and negative values", rewrite_rule="JOIN")
            # All values are negative: concatenate the binary spellings of their absolute values,
            # then return the negated integer value.
            abs_vals = [abs(v) for v in values]
            bits = "".join("0" if v == 0 else format(v, "b") for v in abs_vals)
            return -int(bits or "0", 2)
        # All values non-negative: concatenate their binary spellings in call order.
        bits = "".join("0" if value == 0 else format(value, "b") for value in values)
        return int(bits or "0", 2)
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
        _: List[int],
        __: List[Expression],
        ___: Environment,
        location: SourceLocation,
    ) -> int:
        # Return 1 when the call originates from the primary program file, else 0.
        root = interpreter.entry_filename
        if root == "<string>":
            return 1 if location.file == "<string>" else 0
        return 1 if os.path.abspath(location.file) == root else 0

    def _import(
        self,
        interpreter: "Interpreter",
        _: List[int],
        arg_nodes: List[Expression],
        env: Environment,
        location: SourceLocation,
    ) -> int:
        if len(arg_nodes) != 1 or not isinstance(arg_nodes[0], Identifier):
            raise ASMRuntimeError("IMPORT expects module name identifier", location=location, rewrite_rule="IMPORT")

        module_name = arg_nodes[0].name
        base_dir = os.getcwd() if location.file == "<string>" else os.path.dirname(os.path.abspath(location.file))
        module_path = os.path.join(base_dir, f"{module_name}.asmln")

        try:
            with open(module_path, "r", encoding="utf-8") as handle:
                source_text = handle.read()
        except OSError as exc:
            # Fallback: also search for a `lib` subdirectory adjacent to the
            # interpreter implementation (useful when modules are shipped
            # alongside the interpreter in a `lib/` directory).
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

        interpreter._execute_block(program.statements, env)
        return 0

    def _slice(self, a: int, hi: int, lo: int) -> int:
        if hi < lo:
            raise ASMRuntimeError("SLICE: hi must be >= lo", rewrite_rule="SLICE")
        if lo < 0 or hi < 0:
            raise ASMRuntimeError("SLICE: bit indices must be non-negative", rewrite_rule="SLICE")
        width = hi - lo + 1
        if width <= 0:
            return 0
        # mask off low (hi+1) bits, then shift down by lo
        mask = (1 << (hi + 1)) - 1
        return (a & mask) >> lo
    def _input(
        self,
        interpreter: "Interpreter",
        args: List[int],
        __: List[Expression],
        ___: Environment,
        ____: SourceLocation,
    ) -> int:
        text = interpreter.input_provider()
        interpreter.io_log.append({"event": "INPUT", "text": text})
        if text == "":
            return 0
        if set(text).issubset({"0", "1"}):
            return int(text, 2)
        return 1
    def _print(
        self,
        interpreter: "Interpreter",
        args: List[int],
        __: List[Expression],
        ___: Environment,
        ____: SourceLocation,
    ) -> int:
        text = " ".join(("-" + format(-arg, "b")) if arg < 0 else format(arg, "b") for arg in args)
        interpreter.output_sink(text)
        interpreter.io_log.append({"event": "PRINT", "values": args[:]})
        return 0
    def _assert(
        self,
        interpreter: "Interpreter",
        args: List[int],
        __: List[Expression],
        ___: Environment,
        location: SourceLocation,
    ) -> int:
        if args[0] == 0:
            raise ASMRuntimeError("Assertion failed", location=location, rewrite_rule="ASSERT")
        return 1
    def _delete(
        self,
        interpreter: "Interpreter",
        args: List[int],
        arg_nodes: List[Expression],
        env: Environment,
        location: SourceLocation,
    ) -> int:
        if not arg_nodes or not isinstance(arg_nodes[0], Identifier):
            raise ASMRuntimeError("DEL expects identifier", location=location, rewrite_rule="DEL")
        name = arg_nodes[0].name
        try:
            env.delete(name)
        except ASMRuntimeError as err:
            err.location = location
            raise
        return 0
    def _exist(
        self,
        interpreter: "Interpreter",
        args: List[int],
        arg_nodes: List[Expression],
        env: Environment,
        location: SourceLocation,
    ) -> int:
        # EXIST requires a plain identifier argument and returns 1 if that
        # identifier exists in the current environment (searching parents),
        # otherwise 0. This is intended for LBYL-style checks.
        if not arg_nodes or not isinstance(arg_nodes[0], Identifier):
            raise ASMRuntimeError("EXIST requires an identifier argument", location=location, rewrite_rule="EXIST")
        name = arg_nodes[0].name
        return 1 if env.has(name) else 0
    def _exit(
        self,
        interpreter: "Interpreter",
        args: List[int],
        __: List[Expression],
        ___: Environment,
        ____: SourceLocation,
    ) -> int:
        code = args[0] if args else 0
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
        except ASMRuntimeError as error:
            if self.logger.entries:
                error.step_index = self.logger.entries[-1].step_index
            raise
        else:
            self.call_stack.pop()
    def _execute_block(self, statements: List[Statement], env: Environment) -> None:
        for statement in statements:
            self._execute_statement(statement, env)
    def _execute_statement(self, statement: Statement, env: Environment) -> None:
        self._log_step(rule=statement.__class__.__name__,location=statement.location)
        if(isinstance(statement,Assignment)):
            if(statement.target in self.functions):
                raise ASMRuntimeError(f"Identifier '{statement.target}' already bound as function",location=statement.location,rewrite_rule="ASSIGN")
            value:int = self._evaluate_expression(statement.expression,env)
            env.set(statement.target,value)
            return()
        if(isinstance(statement,ExpressionStatement)):
            self._evaluate_expression(statement.expression,env)
            return()
        if(isinstance(statement,IfStatement)):
            self._execute_if(statement,env)
            return()
        if(isinstance(statement,WhileStatement)):
            self._execute_while(statement,env)
            return()
        if(isinstance(statement,ForStatement)):
            self._execute_for(statement,env)
            return()
        if(isinstance(statement,FuncDef)):
            if(statement.name in self.builtins.table):
                raise ASMRuntimeError(f"Function name '{statement.name}' conflicts with built-in",location=statement.location)
            self.functions[statement.name]=Function(name=statement.name,params=statement.params,body=statement.body,closure=env)
            return()
        if(isinstance(statement,ReturnStatement)):
            frame:Frame = self.call_stack[-1]
            if(frame.name == "<top-level>"):
                raise ASMRuntimeError("RETURN outside of function",location=statement.location,rewrite_rule="RETURN")
            value:int = self._evaluate_expression(statement.expression,env) if statement.expression else 0
            raise ReturnSignal(value)
        raise ASMRuntimeError("Unsupported statement",location=statement.location)
    def _execute_if(self, statement: IfStatement, env: Environment) -> None:
        if(self._evaluate_expression(statement.condition,env) != 0):
            self._execute_block(statement.then_block.statements,env)
            return()
        for branch in statement.elifs:
            if(self._evaluate_expression(branch.condition,env) != 0):
                self._execute_block(branch.block.statements,env)
                return()
        if(statement.else_block):
            self._execute_block(statement.else_block.statements,env)
    def _execute_while(self, statement: WhileStatement, env: Environment) -> None:
        while(self._evaluate_expression(statement.condition,env) != 0):
            self._execute_block(statement.block.statements,env)
    def _execute_for(self, statement: ForStatement, env: Environment) -> None:
        target:int = self._evaluate_expression(statement.target_expr,env)
        env.set(statement.counter,0)
        while(env.get(statement.counter) < target):
            self._execute_block(statement.block.statements,env)
            env.set(statement.counter,env.get(statement.counter) + 1)
    def _evaluate_expression(self, expression: Expression, env: Environment) -> int:
        if(isinstance(expression,Literal)):
            return(expression.value)
        if(isinstance(expression,Identifier)):
            try:
                return(env.get(expression.name))
            except ASMRuntimeError as err:
                err.location = expression.location
                if(expression.name == "INPUT"):
                    result:int = self.builtins.invoke(self,"INPUT",[],[],env,expression.location)
                    self._log_step(rule="INPUT",location=expression.location,extra={"args": [],"result": result})
                    return(result)
                raise
        if(isinstance(expression,CallExpression)):
            if(expression.name == "IMPORT"):
                module_label = expression.args[0].name if (expression.args and isinstance(expression.args[0], Identifier)) else None
                dummy_args:List[int] = [0] * len(expression.args)
                try:
                    result:int = self.builtins.invoke(self,expression.name,dummy_args,expression.args,env,expression.location)
                except ASMRuntimeError:
                    self._log_step(rule="IMPORT",location=expression.location,extra={"module": module_label,"status": "error"})
                    raise
                self._log_step(rule="IMPORT",location=expression.location,extra={"module": module_label,"result": result})
                return(result)

            # Builtins that operate on identifier nodes (DEL, EXIST) must not
            # cause evaluation of the identifier expression (which would raise
            # if the identifier is undefined). Pass dummy numeric args and the
            # raw arg nodes to the builtin so it can inspect names itself.
            if expression.name in ("DEL", "EXIST"):
                dummy_args:List[int] = [0] * len(expression.args)
                try:
                    result:int = self.builtins.invoke(self, expression.name, dummy_args, expression.args, env, expression.location)
                except ASMRuntimeError as err:
                    self._log_step(rule=expression.name, location=expression.location, extra={"args": None, "status": "error"})
                    raise
                self._log_step(rule=expression.name, location=expression.location, extra={"args": None, "result": result})
                return result

            args:List[int] = []
            for arg in expression.args:
                args.append(self._evaluate_expression(arg,env))
            if(expression.name in self.functions):
                self._log_step(rule="CALL",location=expression.location,extra={"function": expression.name,"args": args})
                return(self._call_user_function(self.functions[expression.name],args,expression.location))
            try:
                result:int = self.builtins.invoke(self,expression.name,args,expression.args,env,expression.location)
            except ASMRuntimeError as err:
                self._log_step(rule=expression.name,location=expression.location,extra={"args": args,"status": "error"})
                raise
            self._log_step(rule=expression.name,location=expression.location,extra={"args": args,"result": result})
            return(result)
        raise ASMRuntimeError("Unsupported expression",location=expression.location)
    def _call_user_function(self, function: Function, args: List[int], call_location: SourceLocation) -> int:
        if len(args) != len(function.params):
            raise ASMRuntimeError(
                f"Function {function.name} expects {len(function.params)} arguments but received {len(args)}",
                location=call_location,
            )
        env = Environment(parent=function.closure)
        for param, value in zip(function.params, args):
            env.set(param, value)
        frame = self._new_frame(function.name, env, call_location)
        self.call_stack.append(frame)
        try:
            self._execute_block(function.body.statements, env)
        except ReturnSignal as signal:
            self.call_stack.pop()
            return signal.value
        except ASMRuntimeError:
            raise
        else:
            self.call_stack.pop()
            return 0
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
                lines.append(
                    f"  File \"{frame.location.file}\", line {frame.location.line}, in {frame.name}"
                )
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
def run_cli(argv: Optional[List[str]] = None) -> int:
    def _parse_statements_from_source(text: str, filename: str) -> List[Statement]:
        lexer = Lexer(text, filename)
        tokens = lexer.tokenize()
        parser = Parser(tokens, filename, text.splitlines())
        program = parser.parse()
        return program.statements

    def run_repl(verbose: bool) -> int:
        print("\x1b[38;2;153;221;255mASM-Lang\033[0m REPL. Enter statements, blank line to run buffer.")
        interpreter = Interpreter(source="", filename="<repl>", verbose=verbose)
        global_env = Environment()
        global_frame = interpreter._new_frame("<repl>", global_env, None)
        interpreter.call_stack.append(global_frame)
        buffer: List[str] = []

        while True:
            prompt = "\x1b[38;2;153;221;255m>>>\033[0m " if not buffer else "\x1b[38;2;153;221;255m..>\033[0m "
            try:
                line = input(prompt)
            except EOFError:
                print()
                break

            stripped = line.strip()

            # If buffer is empty, try to execute single-line statements immediately
            # (so EXIT() exits without needing a blank line). Buffer multi-line
            # constructs that start blocks (FUNC, IF, WHILE, FOR) or explicit
            # bracket starts.
            is_block_start = False
            if not buffer:
                uc = stripped.upper()
                if uc.startswith("FUNC") or uc.startswith("IF") or uc.startswith("WHILE") or uc.startswith("FOR"):
                    is_block_start = True
                if stripped.endswith("[") or stripped.endswith("{"):
                    is_block_start = True

            if not buffer and stripped != "" and not is_block_start:
                # try execute this single line immediately
                try:
                    statements = _parse_statements_from_source(line, "<repl>")
                    try:
                        interpreter._execute_block(statements, global_env)
                    except ExitSignal as sig:
                        return sig.code
                except ASMParseError as error:
                    # If parse error, fall back to buffering the line to allow
                    # multi-line input (user may be starting a block).
                    buffer.append(line)
                continue

            if stripped == "" and buffer:
                source_text = "\n".join(buffer)
                buffer.clear()
                try:
                    statements = _parse_statements_from_source(source_text, "<repl>")
                    interpreter._execute_block(statements, global_env)
                except ExitSignal as sig:
                    return sig.code
                except ASMParseError as error:
                    print(f"ParseError: {error}", file=sys.stderr)
                except ASMRuntimeError as error:
                    if interpreter.logger.entries:
                        error.step_index = interpreter.logger.entries[-1].step_index
                    formatter = TracebackFormatter(interpreter)
                    print(formatter.format_text(error, verbose=interpreter.verbose), file=sys.stderr)
                    interpreter.call_stack = [global_frame]
                continue

            buffer.append(line)

        return 0

    parser = argparse.ArgumentParser(description="ASM-Lang reference interpreter")
    parser.add_argument("program", nargs="?", help="Source file path or literal source with -source")
    parser.add_argument("-source", "--source", dest="source_mode", action="store_true", help="Treat program argument as literal source text")
    parser.add_argument("-verbose", "--verbose", dest="verbose", action="store_true", help="Emit env snapshots in tracebacks")
    parser.add_argument("--traceback-json", action="store_true", help="Also emit JSON traceback")
    args = parser.parse_args(argv)

    if args.program is None:
        if args.source_mode:
            print("-source requires a program string", file=sys.stderr)
            return 1
        return run_repl(verbose=args.verbose)

    if args.source_mode:
        source_text = args.program
        filename = "<string>"
    else:
        filename = args.program
        try:
            with open(filename, "r", encoding="utf-8") as handle:
                source_text = handle.read()
        except OSError as exc:
            print(f"Failed to read {filename}: {exc}", file=sys.stderr)
            return 1
    interpreter = Interpreter(source=source_text, filename=filename, verbose=args.verbose)
    try:
        interpreter.run()
    except ASMParseError as error:
        print(f"ParseError: {error}", file=sys.stderr)
        return 1
    except ExitSignal as sig:
        return sig.code
    except ASMRuntimeError as error:
        formatter = TracebackFormatter(interpreter)
        print(formatter.format_text(error, verbose=args.verbose), file=sys.stderr)
        if args.traceback_json:
            print(formatter.to_json(error), file=sys.stderr)
        return 1
    return 0
if __name__ == "__main__":
    raise SystemExit(run_cli())