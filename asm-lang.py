"""ASM-Lang entry point and REPL wiring."""

from __future__ import annotations
import argparse
import sys
from typing import List, Optional

from interpreter import ASMRuntimeError, Environment, ExitSignal, Interpreter, TracebackFormatter
from lexer import ASMParseError, Lexer
from parser import Parser, Statement


def _parse_statements_from_source(text: str, filename: str) -> List[Statement]:
    lexer = Lexer(text, filename)
    tokens = lexer.tokenize()
    parser = Parser(tokens, filename, text.splitlines())
    program = parser.parse()
    return program.statements


def run_repl(verbose: bool) -> int:
    print("\x1b[38;2;153;221;255mASM-Lang\033[0m REPL. Enter statements, blank line to run buffer.") # "ASM-Lang" in light blue
    # Use "<string>" as the REPL's effective source filename so that MAIN() and imports behave
    had_output = False
    def _output_sink(text: str) -> None:
        nonlocal had_output
        had_output = True
        print(text, end="")

    interpreter = Interpreter(source="", filename="<string>", verbose=verbose, input_provider=(lambda: input()), output_sink=_output_sink)
    global_env = Environment()
    # Make the REPL top-level frame mimic script top-level frame
    global_frame = interpreter._new_frame("<top-level>", global_env, None)
    interpreter.call_stack.append(global_frame)
    buffer: List[str] = []

    while True:
        prompt = "\x1b[38;2;153;221;255m>>>\033[0m " if not buffer else "\x1b[38;2;153;221;255m..>\033[0m " # light blue
        if had_output:
            # Ensure prompt starts on a fresh line if the program printed anything
            print()
            had_output = False
        try:
            line = input(prompt)
        except EOFError:
            print()
            break

        stripped = line.strip()

        is_block_start = False
        if not buffer:
            uc = stripped.upper()
            if uc.startswith("FUNC") or uc.startswith("IF") or uc.startswith("WHILE") or uc.startswith("FOR"):
                is_block_start = True
            if stripped.endswith("{"):
                is_block_start = True

        if not buffer and stripped != "" and not is_block_start:
            try:
                statements = _parse_statements_from_source(line, "<string>")
                try:
                    interpreter._execute_block(statements, global_env)
                except ExitSignal as sig:
                    return sig.code
            except ASMParseError:
                # If a single-line parse fails, treat it as start of multi-line input
                buffer.append(line)
            continue

        if stripped == "" and buffer:
            source_text = "\n".join(buffer)
            buffer.clear()
            try:
                statements = _parse_statements_from_source(source_text, "<string>")
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
                # reset call stack to single top-level frame to keep REPL usable
                interpreter.call_stack = [global_frame]
            continue

        buffer.append(line)


def run_cli(argv: Optional[List[str]] = None) -> int:
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