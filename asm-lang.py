"""ASM-Lang entry point and REPL wiring."""

from __future__ import annotations
import argparse
import sys
import os
from typing import List, Optional

from extensions import ASMExtensionError, ReplContext, load_runtime_services
from interpreter import ASMRuntimeError, Environment, ExitSignal, Interpreter, TracebackFormatter
from lexer import ASMParseError, Lexer
from parser import Parser, Statement


def _print_internal_error(*, exc: BaseException, interpreter: Optional[Interpreter] = None, verbose: bool = False, traceback_json: bool = False) -> int:
    """Last-resort error handler.

    Requirement: no ASM-Lang error should ever surface as a Python traceback.
    """
    # Let deliberate exits behave normally.
    if isinstance(exc, SystemExit):
        raise exc
    if isinstance(exc, KeyboardInterrupt):
        print("Interrupted", file=sys.stderr)
        return 130

    message = f"InternalError: {exc.__class__.__name__}: {exc}"
    if interpreter is not None:
        try:
            location = None
            if interpreter.logger.entries:
                location = interpreter.logger.entries[-1].source_location
            error = ASMRuntimeError(message, location=location, rewrite_rule="INTERNAL")
            if interpreter.logger.entries:
                error.step_index = interpreter.logger.entries[-1].step_index
            formatter = TracebackFormatter(interpreter)
            print(formatter.format_text(error, verbose=verbose), file=sys.stderr)
            if traceback_json:
                print(formatter.to_json(error), file=sys.stderr)
            return 1
        except Exception:
            # If formatting itself fails, fall back to a simple one-liner.
            pass

    print(message, file=sys.stderr)
    return 1


def _parse_statements_from_source(text: str, filename: str, *, type_names: Optional[set[str]] = None) -> List[Statement]:
    lexer = Lexer(text, filename)
    tokens = lexer.tokenize()
    parser = Parser(tokens, filename, text.splitlines(), type_names=type_names)
    program = parser.parse()
    return program.statements


def run_repl(*, verbose: bool, services) -> int:
    print("\x1b[38;2;153;221;255mASM-Lang\033[0m REPL. Enter statements, blank line to run buffer.") # "ASM-Lang" in light blue
    # Use "<string>" as the REPL's effective source filename so that MAIN() and imports behave
    had_output = False
    def _output_sink(text: str) -> None:
        nonlocal had_output
        had_output = True
        print(text, end="")

    picked = services.hook_registry.pick_repl() if services is not None else None
    if picked is not None:
        _name, runner, _ext = picked
        ctx = ReplContext(
            verbose=verbose,
            services=services,
            make_interpreter=lambda source, filename: Interpreter(
                source=source,
                filename=filename,
                verbose=verbose,
                services=services,
                input_provider=(lambda: input()),
                output_sink=_output_sink,
            ),
        )
        try:
            return runner(ctx)
        except ASMExtensionError as exc:
            print(f"ExtensionError: {exc}", file=sys.stderr)
            return 1
        except BaseException as exc:
            return _print_internal_error(exc=exc, interpreter=None, verbose=verbose)

    try:
        interpreter = Interpreter(
            source="",
            filename="<string>",
            verbose=verbose,
            services=services,
            input_provider=(lambda: input()),
            output_sink=_output_sink,
        )
    except ASMExtensionError as exc:
        print(f"ExtensionError: {exc}", file=sys.stderr)
        return 1
    except BaseException as exc:
        return _print_internal_error(exc=exc, interpreter=None, verbose=verbose)
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
            if uc.startswith("FUNC") or uc.startswith("IF") or uc.startswith("WHILE") or uc.startswith("FOR") or uc.startswith("PARFOR"):
                is_block_start = True
            if stripped.endswith("{"):
                is_block_start = True

        if not buffer and stripped != "" and not is_block_start:
            try:
                statements = _parse_statements_from_source(line, "<string>", type_names=interpreter.type_registry.names())
                try:
                    interpreter._execute_block(statements, global_env)
                except ExitSignal as sig:
                    return sig.code
                except ASMRuntimeError as error:
                    if interpreter.logger.entries:
                        error.step_index = interpreter.logger.entries[-1].step_index
                    formatter = TracebackFormatter(interpreter)
                    print(formatter.format_text(error, verbose=interpreter.verbose), file=sys.stderr)
                    interpreter.call_stack = [global_frame]
            except ASMParseError:
                # If a single-line parse fails, treat it as start of multi-line input
                buffer.append(line)
            except BaseException as exc:
                _print_internal_error(exc=exc, interpreter=interpreter, verbose=verbose)
            continue

        if stripped == "" and buffer:
            source_text = "\n".join(buffer)
            buffer.clear()
            try:
                statements = _parse_statements_from_source(source_text, "<string>", type_names=interpreter.type_registry.names())
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
            except BaseException as exc:
                _print_internal_error(exc=exc, interpreter=interpreter, verbose=verbose)
                interpreter.call_stack = [global_frame]
            continue

        buffer.append(line)


def run_cli(argv: Optional[List[str]] = None) -> int:
    try:
        parser = argparse.ArgumentParser(description="ASM-Lang reference interpreter")
        parser.add_argument("inputs", nargs="*", help="Program path/source and/or extension files (.py/.asmxt)")
        parser.add_argument("--ext", action="append", default=[], help="Extension path (.py) or pointer file (.asmxt)")
        parser.add_argument("-source", "--source", dest="source_mode", action="store_true", help="Treat program argument as literal source text")
        parser.add_argument("-verbose", "--verbose", dest="verbose", action="store_true", help="Emit env snapshots in tracebacks")
        parser.add_argument("--traceback-json", action="store_true", help="Also emit JSON traceback")
        args = parser.parse_args(argv)
    except SystemExit:
        # argparse uses SystemExit for -h and parse failures; preserve behavior.
        raise
    except BaseException as exc:
        return _print_internal_error(exc=exc, interpreter=None)

    inputs: List[str] = list(args.inputs or [])
    ext_paths: List[str] = list(args.ext or [])
    remaining: List[str] = []
    for item in inputs:
        lower = item.lower()
        if lower.endswith(".py") or lower.endswith(".asmxt"):
            ext_paths.append(item)
        else:
            remaining.append(item)

    # Initialize `program` early so subsequent checks can reference it safely.
    program: Optional[str] = remaining[0] if remaining else None

    # If the caller didn't specify any extensions, look for a pointer file named
    # ".asmxt" in the current working directory or (when a program file was
    # provided) in the program's directory. If found, use it as the extension
    # pointer file so the interpreter loads the extensions it points to.
    if not ext_paths:
        cwd_asmx = os.path.abspath(".asmxt")
        if os.path.exists(cwd_asmx):
            ext_paths.append(cwd_asmx)
        else:
            # If a program path was given (and isn't literal source text),
            # also check the program's directory for a .asmxt pointer file.
            if program and not args.source_mode:
                program_dir = os.path.dirname(os.path.abspath(program))
                program_asmx = os.path.join(program_dir, ".asmxt")
                if os.path.exists(program_asmx):
                    ext_paths.append(program_asmx)
                else:
                    # Also accept a pointer file that shares the program's
                    # basename but uses the .asmxt extension instead of the
                    # program extension (e.g. program.asmln -> program.asmxt).
                    program_alt_asmx = os.path.splitext(os.path.abspath(program))[0] + ".asmxt"
                    if os.path.exists(program_alt_asmx):
                        ext_paths.append(program_alt_asmx)

    try:
        services = load_runtime_services(ext_paths) if ext_paths else load_runtime_services([])
    except ASMExtensionError as exc:
        print(f"ExtensionError: {exc}", file=sys.stderr)
        return 1
    except BaseException as exc:
        return _print_internal_error(exc=exc, interpreter=None, verbose=bool(getattr(args, "verbose", False)))

    program: Optional[str] = None
    if remaining:
        if len(remaining) > 1:
            print("Too many non-extension inputs; expected a single program argument", file=sys.stderr)
            return 1
        program = remaining[0]

    if program is None:
        if args.source_mode and not ext_paths:
            print("-source requires a program string", file=sys.stderr)
            return 1
        # If only extensions are present, run REPL with the loaded extensions.
        return run_repl(verbose=args.verbose, services=services)

    if args.source_mode:
        source_text = program
        filename = "<string>"
    else:
        filename = program
        try:
            with open(filename, "r", encoding="utf-8") as handle:
                source_text = handle.read()
        except OSError as exc:
            print(f"Failed to read {filename}: {exc}", file=sys.stderr)
            return 1

    interpreter: Optional[Interpreter] = None
    try:
        interpreter = Interpreter(source=source_text, filename=filename, verbose=args.verbose, services=services)
    except ASMExtensionError as exc:
        print(f"ExtensionError: {exc}", file=sys.stderr)
        return 1
    except BaseException as exc:
        return _print_internal_error(exc=exc, interpreter=None, verbose=args.verbose, traceback_json=args.traceback_json)

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
    except BaseException as exc:
        return _print_internal_error(exc=exc, interpreter=interpreter, verbose=args.verbose, traceback_json=args.traceback_json)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(run_cli())
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        raise SystemExit(130)
    except SystemExit:
        raise
    except BaseException as exc:
        # Absolute last-resort catch: never print a Python traceback.
        code = _print_internal_error(exc=exc, interpreter=None)
        raise SystemExit(code)