from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple, Union

from lexer import ASMParseError, Token


@dataclass
class SourceLocation:
    file: str
    line: int
    column: int
    statement: str


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
    declared_type: Optional[str]


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
    params: List["Param"]
    return_type: str
    body: Block


@dataclass
class Param:
    type: str
    name: str
    default: Optional["Expression"]


@dataclass
class ReturnStatement(Statement):
    expression: Optional["Expression"]


@dataclass
class PopStatement(Statement):
    expression: "Expression"


@dataclass
class BreakStatement(Statement):
    expression: "Expression"


@dataclass
class GotoStatement(Statement):
    expression: "Expression"


@dataclass
class GotopointStatement(Statement):
    expression: "Expression"


@dataclass
class ContinueStatement(Statement):
    pass


class Expression(Node):
    pass


@dataclass
class TensorLiteral(Expression):
    items: List["Expression"]


@dataclass
class IndexExpression(Expression):
    base: Expression
    indices: List[Expression]


@dataclass
class TensorSetStatement(Statement):
    target: IndexExpression
    value: "Expression"


@dataclass
class Literal(Expression):
    value: Union[int, str]
    literal_type: str


@dataclass
class Identifier(Expression):
    name: str


@dataclass
class CallExpression(Expression):
    name: str
    args: List["CallArgument"]


@dataclass
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
        self.type_names = set(type_names) if type_names is not None else {"INT", "STR", "TNS"}
        self.index = 0

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
        if token.type == "FUNC":
            return self._parse_func()
        if token.type == "IF":
            return self._parse_if()
        if token.type == "WHILE":
            return self._parse_while()
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

    def _parse_typed_assignment(self) -> Assignment:
        type_token = self._consume_type_token()
        self._consume("COLON")
        return self._parse_assignment(type_token.value)

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

    def _parse_if(self) -> IfStatement:
        keyword = self._consume("IF")
        condition: Expression = self._parse_parenthesized_expression()
        then_block: Block = self._parse_block()
        elifs: List[IfBranch] = []
        while self._match("ELSIF"):
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
        while self._peek().type == "LBRACKET":
            expr = self._parse_index_suffix(expr)
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
        if token.type == "STRING":
            string_token = self._consume("STRING")
            return Literal(location=self._location_from_token(string_token), value=string_token.value, literal_type="STR")
        if token.type == "LBRACKET":
            return self._parse_tensor_literal()
        if token.type == "IDENT":
            ident: Token = self._consume("IDENT")
            location: SourceLocation = self._location_from_token(ident)
            if self._match("LPAREN"):
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
                return CallExpression(location=location, name=ident.value, args=args)
            return Identifier(location=location, name=ident.value)
        if token.type == "LPAREN":
            self._consume("LPAREN")
            expr: Expression = self._parse_expression()
            self._consume("RPAREN")
            return expr
        raise ASMParseError(f"Unexpected token {token.type} in expression at line {token.line}")

    def _parse_index_suffix(self, base: Expression) -> IndexExpression:
        lbracket = self._consume("LBRACKET")
        indices: List[Expression] = []
        if self._peek().type != "RBRACKET":
            while True:
                indices.append(self._parse_expression())
                if not self._match("COMMA"):
                    break
        self._consume("RBRACKET")
        return IndexExpression(location=self._location_from_token(lbracket), base=base, indices=indices)

    def _parse_index_expression(self) -> IndexExpression:
        expr = self._parse_primary()
        if self._peek().type != "LBRACKET":
            raise ASMParseError(f"Expected '[' in indexed assignment at line {self._peek().line}")
        expr = self._parse_index_suffix(expr)
        while self._peek().type == "LBRACKET":
            expr = self._parse_index_suffix(expr)
        if not isinstance(expr, IndexExpression):
            raise ASMParseError(f"Invalid indexed assignment at line {self._peek().line}")
        return expr

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
        if i >= len(tokens) or tokens[i].type != "LBRACKET":
            return False

        # Walk balanced brackets to find the end of the indexed target.
        depth = 0
        while i < len(tokens):
            tok = tokens[i]
            if tok.type == "LBRACKET":
                depth += 1
            elif tok.type == "RBRACKET":
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
                if tok.type == "LBRACKET":
                    depth += 1
                elif tok.type == "RBRACKET":
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
        token = self._consume("IDENT")
        if token.value not in self.type_names:
            raise ASMParseError(f"Unknown type '{token.value}' at line {token.line}")
        return token

    def _is_typed_assignment_start(self) -> bool:
        current = self._peek()
        if current.type != "IDENT" or current.value not in self.type_names:
            return False
        if self.index + 1 >= len(self.tokens):
            return False
        return self._peek_next().type == "COLON"
