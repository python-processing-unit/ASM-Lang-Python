from __future__ import annotations
from dataclasses import dataclass
from typing import List


class ASMError(Exception):
    """Base class for interpreter errors."""


class ASMParseError(ASMError):
    """Raised when parsing fails."""


@dataclass
class Token:
    type: str
    value: str
    line: int
    column: int


KEYWORDS = {
    "IF",
    "ELSIF",
    "ELSE",
    "WHILE",
    "FOR",
    "FUNC",
    "RETURN",
    "BREAK",
    "CONTINUE",
    "GOTO",
    "GOTOPOINT"
}

SYMBOLS = {
    "(": "LPAREN",
    ")": "RPAREN",
    "[": "LBRACKET",
    "]": "RBRACKET",
    "{": "LBRACE",
    "}": "RBRACE",
    ",": "COMMA",
    "=": "EQUALS",
    ":": "COLON",
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
        _peek = self._peek
        _advance = self._advance
        _is_identifier_start = self._is_identifier_start
        symbols = SYMBOLS
        while not self._eof:
            ch: str = _peek()
            if ch in " \t":
                _advance()
                continue
            if ch == "\r":
                _advance()
                continue
            if ch == "^":
                self._consume_line_continuation()
                continue
            if ch == "\n":
                tokens.append(Token("NEWLINE", "\n", self.line, self.column))
                _advance()
                continue
            if ch == "#":
                self._consume_comment()
                continue
            if ch in SYMBOLS:
                tokens.append(Token(symbols[ch], ch, self.line, self.column))
                _advance()
                continue
            if ch in ('"', "'"):
                tokens.append(self._consume_string())
                continue
            if ch == "-":
                tokens.append(self._consume_signed_number())
                continue
            if ch in "01":
                tokens.append(self._consume_unsigned_number())
                continue
            if _is_identifier_start(ch):
                tokens.append(self._consume_identifier())
                continue
            raise ASMParseError(
                f"Unexpected character '{ch}' at {self.filename}:{self.line}:{self.column}"
            )
        tokens.append(Token("EOF", "", self.line, self.column))
        return tokens

    def _consume_comment(self) -> None:
        while not self._eof and self._peek() != "\n":
            self._advance()

    def _consume_unsigned_number(self) -> Token:
        line, col = self.line, self.column
        digits = self._consume_binary_digits()
        return Token("NUMBER", digits, line, col)

    def _consume_string(self) -> Token:
        line, col = self.line, self.column
        opening = self._peek()
        if opening not in ('"', "'"):
            raise ASMParseError(
                f"Expected string opening quote at {self.filename}:{line}:{col}"
            )
        self._advance()  # consume opening quote
        chars: List[str] = []
        while not self._eof:
            ch = self._peek()
            if ch == opening:
                self._advance()
                return Token("STRING", "".join(chars), line, col)
            if ch == "\n":
                raise ASMParseError(
                    f"Unterminated string literal at {self.filename}:{line}:{col}"
                )
            chars.append(ch)
            self._advance()
        raise ASMParseError(
            f"Unterminated string literal at {self.filename}:{line}:{col}"
        )

    def _consume_signed_number(self) -> Token:
        line, col = self.line, self.column
        self._advance()
        while not self._eof and self._peek() in " \t\r":
            self._advance()
        if self._eof or self._peek() not in "01":
            raise ASMParseError(f"Expected binary digits after '-' at {self.filename}:{line}:{col}")
        digits = self._consume_binary_digits()
        return Token("NUMBER", "-" + digits, line, col)

    def _consume_binary_digits(self) -> str:
        digits: List[str] = []
        while not self._eof:
            ch = self._peek()
            if ch in "01":
                digits.append(ch)
                self._advance()
                continue
            if ch == "^":
                self._consume_line_continuation()
                continue
            break
        return "".join(digits)

    def _consume_identifier(self) -> Token:
        line, col = self.line, self.column
        if not self._eof and self._peek() in "01":
            raise ASMParseError(
                f"Identifiers must not start with '0' or '1' at {self.filename}:{line}:{col}"
            )
        chars: List[str] = []
        while not self._eof:
            ch = self._peek()
            if self._is_identifier_part(ch):
                chars.append(ch)
                self._advance()
                continue
            if ch == "^":
                self._consume_line_continuation()
                continue
            break
        value = "".join(chars)
        token_type: str = value if value in KEYWORDS else "IDENT"
        return Token(token_type, value, line, col)

    def _is_identifier_start(self, ch: str) -> bool:
        return (ch in "abcdefghijklmnopqrstuvwxyz23456789;/ABCDEFGHIFJKLMNOPQRSTUVWXYZ!@$%&~_+|<>?")

    def _is_identifier_part(self, ch: str) -> bool:
        return (ch in "abcdefghijklmnopqrstuvwxyz1234567890;./ABCDEFGHIFJKLMNOPQRSTUVWXYZ!@$%&~_+|<>?")
        # "." is not actually a valid character in namespace symbols, but is allowed since it is used to separate module names from namespace symbols.

    def _consume_line_continuation(self) -> None:
        if self.index + 1 >= len(self.text):
            raise ASMParseError(
                f"Invalid line continuation '^' at {self.filename}:{self.line}:{self.column}"
            )
        ch1 = self.text[self.index + 1]
        if ch1 == "\n":
            self._advance()
            self._advance()
            return
        if ch1 == "\r":
            self._advance()
            self._advance()
            if self.index < len(self.text) and self.text[self.index] == "\n":
                self._advance()
            return
        raise ASMParseError(
            f"Invalid line continuation '^' not followed by newline at {self.filename}:{self.line}:{self.column}"
        )

    @property
    def _eof(self) -> bool:
        return self.index >= len(self.text)

    def _peek(self) -> str:
        return self.text[self.index]

    def _advance(self) -> None:
        if self.text[self.index] == "\n":
            self.line += 1
            self.column = 1
        else:
            self.column += 1
        self.index += 1
