from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple, Union

from fractions import Fraction

from lexer import ASMParseError, Token


@dataclass(slots=True)
class SourceLocation:
    file: str
    line: int
    column: int
    statement: str


@dataclass(slots=True)
class Node:
    location: SourceLocation


@dataclass(slots=True)
class Program(Node):
    statements: List["Statement"]


class Statement(Node):
    pass


@dataclass(slots=True)
class Block(Node):
    statements: List[Statement]


@dataclass(slots=True)
class Assignment(Statement):
    target: str
    declared_type: Optional[str]

    expression: "Expression"


@dataclass(slots=True)
class Declaration(Statement):
    name: str
    declared_type: str


@dataclass(slots=True)
class ExpressionStatement(Statement):
    expression: "Expression"


@dataclass(slots=True)
class IfBranch:
    condition: "Expression"
    block: Block


@dataclass(slots=True)
class IfStatement(Statement):
    condition: "Expression"
    then_block: Block
    elifs: List[IfBranch]
    else_block: Optional[Block]


@dataclass(slots=True)
class WhileStatement(Statement):
    condition: "Expression"
    block: Block


@dataclass(slots=True)
class ForStatement(Statement):
    counter: str
    target_expr: "Expression"
    block: Block


@dataclass(slots=True)
class ParForStatement(Statement):
    counter: str
    target_expr: "Expression"
    block: Block


@dataclass(slots=True)
class FuncDef(Statement):
    name: str
    params: List["Param"]
    return_type: str
    body: Block


@dataclass(slots=True)
class Param:
    type: str
    name: str
    default: Optional["Expression"]


@dataclass(slots=True)
class ReturnStatement(Statement):
    expression: Optional["Expression"]


@dataclass(slots=True)
class PopStatement(Statement):
    expression: "Expression"


@dataclass(slots=True)
class BreakStatement(Statement):
    expression: "Expression"


@dataclass(slots=True)
class GotoStatement(Statement):
    expression: "Expression"


@dataclass(slots=True)
class GotopointStatement(Statement):
    expression: "Expression"


@dataclass(slots=True)
class ContinueStatement(Statement):
    pass


@dataclass(slots=True)
class AsyncStatement(Statement):
    block: Block


@dataclass(slots=True)
class TryStatement(Statement):
    try_block: Block
    catch_symbol: Optional[str]
    catch_block: Block


class Expression(Node):
    pass


@dataclass(slots=True)
class LambdaExpression(Expression):
    params: List["Param"]
    return_type: str
    body: Block


@dataclass(slots=True)
class TensorLiteral(Expression):
    items: List["Expression"]


@dataclass(slots=True)
class MapLiteral(Expression):
    items: List[Tuple["Expression", "Expression"]]


@dataclass(slots=True)
class IndexExpression(Expression):
    base: Expression
    indices: List[Expression]


@dataclass(slots=True)
class Range(Expression):
    lo: Expression
    hi: Expression


@dataclass(slots=True)
class Star(Expression):
    """Represents a full-dimension slice `*` used inside tensor indices."""
    pass


@dataclass(slots=True)
class TensorSetStatement(Statement):
    target: IndexExpression
    value: "Expression"


@dataclass(slots=True)
class Literal(Expression):
    value: Union[int, float, str]
    literal_type: str


@dataclass(slots=True)
class Identifier(Expression):
    name: str


@dataclass(slots=True)
class PointerExpression(Expression):
    target: str


@dataclass(slots=True)
class CallExpression(Expression):
    callee: Expression
    args: List["CallArgument"]


@dataclass(slots=True)
class CallArgument:
    name: Optional[str]
    expression: Expression


class Parser:
    def __init__(
        self,
        tokens: List[Token],
        filename: str,
        source_lines: List[str],
        *,
        type_names: Optional[Iterable[str]] = None,
    ):
        self.tokens = tokens
        self.filename = filename
        self.source_lines = source_lines
        self.type_names = set(type_names) if type_names is not None else {"INT", "FLT", "STR", "TNS", "FUNC", "MAP"}
        self.index = 0

    def _parse_flt_literal(self, raw: str, *, token: Token) -> float:
        # raw is of the form [-]?[01]+\.[01]+
        text = raw.strip()
        neg = text.startswith("-")
        if neg:
            text = text[1:]
        if "." not in text:
            raise ASMParseError(f"Invalid FLT literal at line {token.line}")
        left, right = text.split(".", 1)
        if left == "" or right == "":
            raise ASMParseError(f"Invalid FLT literal at line {token.line}")
        if not set(left).issubset({"0", "1"}) or not set(right).issubset({"0", "1"}):
            raise ASMParseError(f"Invalid FLT literal at line {token.line}")
        numerator = (int(left, 2) << len(right)) + int(right, 2)
        denom = 1 << len(right)
        value = float(Fraction(numerator, denom))
        return -value if neg else value

    def parse(self) -> Program:
        statements: List[Statement] = self._parse_statements(stop_tokens={"EOF"})
        eof_token: Token = self._peek()
        return Program(location=self._location_from_token(eof_token), statements=statements)

    def _parse_statements(self, stop_tokens: Iterable[str]) -> List[Statement]:
        statements: List[Statement] = []
        while self._peek().type not in stop_tokens:
            if self._match("NEWLINE"):
                continue
            statements.append(self._parse_statement())
            self._consume_newlines()
        return statements

    def _parse_statement(self) -> Statement:
        if self._is_typed_assignment_start():
            return self._parse_typed_assignment()
        token = self._peek()
        if self._looks_like_index_assignment():
            return self._parse_index_assignment()
        if token.type == "TRY":
            return self._parse_try()
        if token.type == "CATCH":
            raise ASMParseError(f"CATCH must immediately follow a TRY block at line {token.line}")
        if token.type == "FUNC":
            return self._parse_func()
        if token.type == "IF":
            return self._parse_if()
        if token.type == "ASYNC":
            return self._parse_async()
        if token.type == "WHILE":
            return self._parse_while()
        if token.type == "PARFOR":
            return self._parse_parfor()
        if token.type == "FOR":
            return self._parse_for()
        if token.type == "RETURN":
            return self._parse_return()
        if token.type == "POP":
            return self._parse_pop()
        if token.type == "BREAK":
            return self._parse_break()
        if token.type == "CONTINUE":
            return self._parse_continue()
        if token.type == "GOTO":
            return self._parse_goto()
        if token.type == "GOTOPOINT":
            return self._parse_gotopoint()
        if token.type == "IDENT" and self._peek_next().type == "EQUALS":
            return self._parse_assignment(None)
        expr: Expression = self._parse_expression()
        return ExpressionStatement(location=expr.location, expression=expr)

    def _parse_assignment(self, declared_type: Optional[str]) -> Assignment:
        ident = self._consume("IDENT")
        self._consume("EQUALS")
        expr = self._parse_expression()
        location: SourceLocation = self._location_from_token(ident)
        return Assignment(location=location, target=ident.value, expression=expr, declared_type=declared_type)

    def _parse_typed_assignment(self) -> Statement:
        type_token = self._consume_type_token()
        self._consume("COLON")
        # Allow a bare type declaration (e.g. `INT: x`) which records the
        # symbol's declared type but does not perform an assignment. If an
        # equals follows, parse as a normal typed assignment.
        if self._peek().type == "IDENT" and self._peek_next().type == "EQUALS":
            return self._parse_assignment(type_token.value)
        ident = self._consume("IDENT")
        location: SourceLocation = self._location_from_token(type_token)
        return Declaration(location=location, name=ident.value, declared_type=type_token.value)

    def _parse_index_assignment(self) -> TensorSetStatement:
        target = self._parse_index_expression()
        equals_token = self._consume("EQUALS")
        value_expr = self._parse_expression()
        return TensorSetStatement(location=self._location_from_token(equals_token), target=target, value=value_expr)

    def _parse_func(self) -> FuncDef:
        keyword = self._consume("FUNC")
        name_token = self._consume("IDENT")
        self._consume("LPAREN")
        params: List[Param] = []
        seen_default = False
        if self._peek().type != "RPAREN":
            while True:
                type_token = self._consume_type_token()
                self._consume("COLON")
                name_tok = self._consume("IDENT")
                default_expr: Optional[Expression] = None
                if self._match("EQUALS"):
                    seen_default = True
                    default_expr = self._parse_expression()
                elif seen_default:
                    raise ASMParseError(
                        f"Positional parameter cannot follow parameter with default at line {name_tok.line}")
                params.append(Param(type=type_token.value, name=name_tok.value, default=default_expr))
                if not self._match("COMMA"):
                    break
        self._consume("RPAREN")
        self._consume("COLON")
        return_type = self._consume_type_token()
        block: Block = self._parse_block()
        location: SourceLocation = self._location_from_token(keyword)
        return FuncDef(location=location, name=name_token.value, params=params, return_type=return_type.value, body=block)

    def _parse_lambda(self) -> LambdaExpression:
        keyword = self._consume("LAMBDA")
        self._consume("LPAREN")
        params: List[Param] = []
        seen_default = False
        if self._peek().type != "RPAREN":
            while True:
                type_token = self._consume_type_token()
                self._consume("COLON")
                name_tok = self._consume("IDENT")
                default_expr: Optional[Expression] = None
                if self._match("EQUALS"):
                    seen_default = True
                    default_expr = self._parse_expression()
                elif seen_default:
                    raise ASMParseError(
                        f"Positional parameter cannot follow parameter with default at line {name_tok.line}")
                params.append(Param(type=type_token.value, name=name_tok.value, default=default_expr))
                if not self._match("COMMA"):
                    break
        self._consume("RPAREN")
        self._consume("COLON")
        return_type = self._consume_type_token()
        block: Block = self._parse_block()
        location: SourceLocation = self._location_from_token(keyword)
        return LambdaExpression(location=location, params=params, return_type=return_type.value, body=block)

    def _parse_if(self) -> IfStatement:
        keyword = self._consume("IF")
        condition: Expression = self._parse_parenthesized_expression()
        then_block: Block = self._parse_block()
        elifs: List[IfBranch] = []
        while self._match("ELSEIF"):
            cond: Expression = self._parse_parenthesized_expression()
            block: Block = self._parse_block()
            elifs.append(IfBranch(condition=cond, block=block))
        else_block: Optional[Block] = self._parse_block() if self._match("ELSE") else None
        return IfStatement(
            location=self._location_from_token(keyword),
            condition=condition,
            then_block=then_block,
            elifs=elifs,
            else_block=else_block,
        )

    def _parse_while(self) -> WhileStatement:
        keyword = self._consume("WHILE")
        condition: Expression = self._parse_parenthesized_expression()
        block: Block = self._parse_block()
        return WhileStatement(location=self._location_from_token(keyword), condition=condition, block=block)

    def _parse_for(self) -> ForStatement:
        keyword = self._consume("FOR")
        self._consume("LPAREN")
        counter = self._consume("IDENT")
        self._consume("COMMA")
        target: Expression = self._parse_expression()
        self._consume("RPAREN")
        block: Block = self._parse_block()
        return ForStatement(location=self._location_from_token(keyword), counter=counter.value, target_expr=target, block=block)

    def _parse_parfor(self) -> ParForStatement:
        keyword = self._consume("PARFOR")
        self._consume("LPAREN")
        counter = self._consume("IDENT")
        self._consume("COMMA")
        target: Expression = self._parse_expression()
        self._consume("RPAREN")
        block: Block = self._parse_block()
        return ParForStatement(location=self._location_from_token(keyword), counter=counter.value, target_expr=target, block=block)

    def _parse_return(self) -> ReturnStatement:
        keyword = self._consume("RETURN")
        expression: Expression = self._parse_parenthesized_expression()
        return ReturnStatement(location=self._location_from_token(keyword), expression=expression)

    def _parse_pop(self) -> PopStatement:
        keyword = self._consume("POP")
        expression: Expression = self._parse_parenthesized_expression()
        return PopStatement(location=self._location_from_token(keyword), expression=expression)

    def _parse_break(self) -> BreakStatement:
        keyword = self._consume("BREAK")
        expression: Expression = self._parse_parenthesized_expression()
        return BreakStatement(location=self._location_from_token(keyword), expression=expression)

    def _parse_goto(self) -> GotoStatement:
        keyword = self._consume("GOTO")
        expression: Expression = self._parse_parenthesized_expression()
        return GotoStatement(location=self._location_from_token(keyword), expression=expression)

    def _parse_gotopoint(self) -> GotopointStatement:
        keyword = self._consume("GOTOPOINT")
        expression: Expression = self._parse_parenthesized_expression()
        return GotopointStatement(location=self._location_from_token(keyword), expression=expression)

    def _parse_continue(self) -> ContinueStatement:
        keyword = self._consume("CONTINUE")
        # Expect empty parentheses: CONTINUE()
        self._consume("LPAREN")
        self._consume("RPAREN")
        return ContinueStatement(location=self._location_from_token(keyword))

    def _parse_async(self) -> AsyncStatement:
        keyword = self._consume("ASYNC")
        block: Block = self._parse_block()
        return AsyncStatement(location=self._location_from_token(keyword), block=block)

    def _parse_try(self) -> TryStatement:
        keyword = self._consume("TRY")
        try_block: Block = self._parse_block()
        # Allow newlines between the TRY block and its required CATCH.
        self._consume_newlines()
        if self._peek().type != "CATCH":
            tok = self._peek()
            raise ASMParseError(
                f"TRY must be followed by a CATCH block (found {tok.type}) at line {tok.line}"
            )
        self._consume("CATCH")
        catch_symbol: Optional[str] = None
        if self._match("LPAREN"):
            # Syntax: CATCH(sym){...} or CATCH() for no symbol.
            if self._peek().type == "RPAREN":
                # Empty parentheses: CATCH()
                self._consume("RPAREN")
            elif self._peek().type == "IDENT":
                # Single identifier: CATCH(name)
                sym = self._consume("IDENT")
                self._consume("RPAREN")
                catch_symbol = sym.value
            else:
                tok = self._peek()
                raise ASMParseError(
                    f"Invalid CATCH syntax (found {tok.type}) at line {tok.line}"
                )
        catch_block: Block = self._parse_block()
        return TryStatement(
            location=self._location_from_token(keyword),
            try_block=try_block,
            catch_symbol=catch_symbol,
            catch_block=catch_block,
        )

    def _parse_block(self) -> Block:
        opening = self._peek().type
        if opening == "LBRACE":
            start = self._consume("LBRACE")
            closing = "RBRACE"
        else:
            raise ASMParseError(f"Expected '{{' to start block but found {opening}")
        statements: List[Statement] = self._parse_statements(stop_tokens={closing})
        self._consume(closing)
        return Block(location=self._location_from_token(start), statements=statements)

    def _parse_expression(self) -> Expression:
        expr = self._parse_primary()
        while True:
            tok_type = self._peek().type
            if tok_type == "LBRACKET" or tok_type == "LANGLE":
                expr = self._parse_index_suffix(expr)
                continue
            if tok_type == "LPAREN":
                expr = self._parse_call_suffix(expr)
                continue
            break
        return expr

    def _parse_primary(self) -> Expression:
        token = self._peek()
        if token.type == "NUMBER":
            number: Token = self._consume("NUMBER")
            sval = number.value
            if sval.startswith("-"):
                if len(sval) == 1:
                    raise ASMParseError(f"Invalid numeric literal at line {number.line}")
                value = -int(sval[1:], 2)
            else:
                value = int(sval, 2)
            return Literal(location=self._location_from_token(number), value=value, literal_type="INT")
        if token.type == "FLOAT":
            flt: Token = self._consume("FLOAT")
            value = self._parse_flt_literal(flt.value, token=flt)
            return Literal(location=self._location_from_token(flt), value=value, literal_type="FLT")
        if token.type == "STRING":
            string_token = self._consume("STRING")
            return Literal(location=self._location_from_token(string_token), value=string_token.value, literal_type="STR")
        if token.type == "LBRACKET":
            return self._parse_tensor_literal()
        if token.type == "LANGLE":
            return self._parse_map_literal()
        if token.type == "LAMBDA":
            return self._parse_lambda()
        if token.type == "IDENT":
            ident: Token = self._consume("IDENT")
            location: SourceLocation = self._location_from_token(ident)
            return Identifier(location=location, name=ident.value)
        if token.type == "AT":
            at_tok = self._consume("AT")
            ident_tok = self._consume("IDENT")
            location: SourceLocation = self._location_from_token(at_tok)
            return PointerExpression(location=location, target=ident_tok.value)
        if token.type == "LPAREN":
            self._consume("LPAREN")
            expr: Expression = self._parse_expression()
            self._consume("RPAREN")
            return expr
        raise ASMParseError(f"Unexpected token {token.type} in expression at line {token.line}")

    def _parse_index_suffix(self, base: Expression) -> IndexExpression:
        start_tok = self._peek()
        if start_tok.type == "LBRACKET":
            lbracket = self._consume("LBRACKET")
            closing = "RBRACKET"
        else:
            lbracket = self._consume("LANGLE")
            closing = "RANGLE"
        indices: List[Expression] = []
        if self._peek().type != closing:
            while True:
                # Support star `*` (full-dimension slice), and slice
                # syntax lo - hi inside index brackets. The lexer emits a
                # DASH token for '-' when it's not starting a signed number.
                if self._peek().type == "STAR":
                    star_tok = self._consume("STAR")
                    indices.append(Star(location=self._location_from_token(star_tok)))
                else:
                    first = self._parse_expression()
                    if self._peek().type == "DASH":
                        self._consume("DASH")
                        second = self._parse_expression()
                        indices.append(Range(lo=first, hi=second, location=first.location))
                    else:
                        indices.append(first)
                if not self._match("COMMA"):
                    break
        self._consume(closing)
        return IndexExpression(location=self._location_from_token(lbracket), base=base, indices=indices)

    def _parse_index_expression(self) -> IndexExpression:
        expr = self._parse_primary()
        if self._peek().type not in ("LBRACKET", "LANGLE"):
            raise ASMParseError(f"Expected '[' or '<' in indexed assignment at line {self._peek().line}")
        expr = self._parse_index_suffix(expr)
        while self._peek().type in ("LBRACKET", "LANGLE"):
            expr = self._parse_index_suffix(expr)
        if not isinstance(expr, IndexExpression):
            raise ASMParseError(f"Invalid indexed assignment at line {self._peek().line}")
        return expr

    def _parse_call_suffix(self, callee: Expression) -> CallExpression:
        lparen = self._consume("LPAREN")
        args: List[CallArgument] = []
        seen_kw = False
        if self._peek().type != "RPAREN":
            while True:
                if self._peek().type == "IDENT" and self._peek_next().type == "EQUALS":
                    name_tok = self._consume("IDENT")
                    self._consume("EQUALS")
                    arg_expr = self._parse_expression()
                    seen_kw = True
                    args.append(CallArgument(name=name_tok.value, expression=arg_expr))
                else:
                    if seen_kw:
                        raise ASMParseError(
                            f"Positional argument cannot follow keyword argument at line {self._peek().line}")
                    args.append(CallArgument(name=None, expression=self._parse_expression()))
                if not self._match("COMMA"):
                    break
        self._consume("RPAREN")
        return CallExpression(location=self._location_from_token(lparen), callee=callee, args=args)

    def _parse_tensor_literal(self) -> TensorLiteral:
        lbracket = self._consume("LBRACKET")
        items: List[Expression] = []
        if self._peek().type != "RBRACKET":
            while True:
                items.append(self._parse_expression())
                if not self._match("COMMA"):
                    break
        self._consume("RBRACKET")
        return TensorLiteral(location=self._location_from_token(lbracket), items=items)

    def _parse_map_literal(self) -> MapLiteral:
        langle = self._consume("LANGLE")
        items: List[Tuple[Expression, Expression]] = []
        if self._peek().type != "RANGLE":
            while True:
                key_expr = self._parse_expression()
                self._consume("EQUALS")
                val_expr = self._parse_expression()
                items.append((key_expr, val_expr))
                if not self._match("COMMA"):
                    break
        self._consume("RANGLE")
        return MapLiteral(location=self._location_from_token(langle), items=items)

    def _parse_parenthesized_expression(self) -> Expression:
        self._consume("LPAREN")
        expr = self._parse_expression()
        self._consume("RPAREN")
        return expr

    def _looks_like_index_assignment(self) -> bool:
        i = self.index
        tokens = self.tokens
        if i >= len(tokens) or tokens[i].type != "IDENT":
            return False
        i += 1
        if i >= len(tokens) or tokens[i].type not in ("LBRACKET", "LANGLE"):
            return False

        # Walk balanced brackets to find the end of the indexed target.
        depth = 0
        while i < len(tokens):
            tok = tokens[i]
            if tok.type in ("LBRACKET", "LANGLE"):
                depth += 1
            elif tok.type in ("RBRACKET", "RANGLE"):
                depth -= 1
                if depth == 0:
                    i += 1
                    break
            i += 1
        if depth != 0:
            return False

        # Support additional bracketed suffixes like a[1][2].
        while i < len(tokens) and tokens[i].type == "LBRACKET":
            depth = 0
            i += 1
            while i < len(tokens):
                tok = tokens[i]
                if tok.type in ("LBRACKET", "LANGLE"):
                    depth += 1
                elif tok.type in ("RBRACKET", "RANGLE"):
                    depth -= 1
                    if depth == 0:
                        i += 1
                        break
                i += 1
            if depth != 0:
                return False

        while i < len(tokens) and tokens[i].type == "NEWLINE":
            i += 1
        return i < len(tokens) and tokens[i].type == "EQUALS"

    def _consume(self, token_type: str) -> Token:
        token = self._peek()
        if token.type != token_type:
            raise ASMParseError(f"Expected token {token_type} but found {token.type} at line {token.line}")
        self.index += 1
        return token

    def _match(self, token_type: str) -> bool:
        if self._peek().type == token_type:
            self.index += 1
            return True
        return False

    def _consume_newlines(self) -> None:
        while self._match("NEWLINE"):
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
        return SourceLocation(file=self.filename, line=token.line, column=token.column, statement=statement)

    def _consume_type_token(self) -> Token:
        token = self._peek()
        if token.value not in self.type_names:
            raise ASMParseError(f"Unknown type '{token.value}' at line {token.line}")
        self.index += 1
        return token

    def _is_typed_assignment_start(self) -> bool:
        current = self._peek()
        if current.value not in self.type_names:
            return False
        if self.index + 1 >= len(self.tokens):
            return False
        return self._peek_next().type == "COLON"
