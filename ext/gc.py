"""ASM-Lang Extension: cautious garbage collector.

Behavior:
- Deletes bindings only when there are no future references in the remaining
  executable statements (conservative).
- Never deletes bindings in the REPL.
- Provides GCIGNORE(symbol) to permanently exclude a symbol from GC.

This extension intentionally errs on the side of *not* collecting.
"""

from __future__ import annotations

from typing import Dict, FrozenSet, Iterable, List, Optional, Set, Tuple

from extensions import ExtensionAPI


ASM_LANG_EXTENSION_NAME = "gc"
ASM_LANG_EXTENSION_API_VERSION = 1


# ---- Interpreter monkeypatching ----


class _BlockCtx:
    __slots__ = ("statements", "index", "in_loop")

    def __init__(self, *, statements: List[object], index: int = -1, in_loop: bool = False) -> None:
        self.statements = statements
        self.index = index
        self.in_loop = in_loop


class _GCState:
    __slots__ = (
        "ignored",
        "block_stack",
        "loop_depth",
        "goto_seen_in_frame",
        "block_analysis_cache",
        "function_epoch",
        "closure_refs_cache",
        "function_analysis_cache",
    )

    def __init__(self) -> None:
        self.ignored: Set[str] = set()
        self.block_stack: List[_BlockCtx] = []
        self.loop_depth: int = 0
        self.goto_seen_in_frame: bool = False
        # Cache: id(statements_list) -> _BlockAnalysis (keeps a strong ref to the list)
        self.block_analysis_cache: Dict[int, _BlockAnalysis] = {}
        # Bumped whenever a FuncDef executes (module import included)
        self.function_epoch: int = 0
        # Cache: (id(env), function_epoch) -> (refs, prefixes)
        self.closure_refs_cache: Dict[Tuple[int, int], Tuple[Set[str], Set[str]]] = {}
        # Cache: fn_name -> (id(fn.body), refs, prefixes, calls)
        self.function_analysis_cache: Dict[str, Tuple[int, Set[str], Set[str], Set[str]]] = {}


def _get_state(interpreter) -> _GCState:
    state = getattr(interpreter, "_gc_ext_state", None)
    if state is None:
        state = _GCState()
        setattr(interpreter, "_gc_ext_state", state)
    return state


def _is_repl_interpreter(interpreter) -> bool:
    # In the REPL, asm-lang.py constructs a single Interpreter with:
    # - filename "<string>"
    # - source "" (statements are parsed externally per input buffer)
    return getattr(interpreter, "filename", None) == "<string>" and getattr(interpreter, "source", None) == ""


def _patch_interpreter_once() -> None:
    # Import lazily to avoid any import-time surprises.
    from interpreter import Interpreter

    if getattr(Interpreter, "_gc_ext_patched", False):
        return

    Interpreter._gc_ext_patched = True  # type: ignore[attr-defined]

    # Save originals
    _orig_execute_block = Interpreter._execute_block
    _orig_execute_while = Interpreter._execute_while
    _orig_execute_for = Interpreter._execute_for

    def _execute_block_wrapped(self, statements, env):
        state = _get_state(self)
        # Preserve list identity when possible so caches work across loop iterations.
        statements_list = statements if isinstance(statements, list) else list(statements)
        state.block_stack.append(_BlockCtx(statements=statements_list, index=-1, in_loop=state.loop_depth > 0))
        try:
            # Re-implement the loop so we can track the current index.
            i = 0
            frame = self.call_stack[-1]
            gotopoints = frame.gotopoints
            while i < len(statements):
                state.block_stack[-1].index = i
                statement = statements[i]
                self._emit_event("before_statement", self, statement, env)
                # Preserve special gotopoint bookkeeping behavior.
                from parser import GotopointStatement

                if isinstance(statement, GotopointStatement):
                    self._log_step(rule=statement.__class__.__name__, location=statement.location)
                    gid = self._evaluate_expression(statement.expression, env)
                    from interpreter import TYPE_INT, TYPE_STR, ASMRuntimeError

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
                    self._emit_event("after_statement", self, statement, env)
                    continue

                try:
                    self._execute_statement(statement, env)
                except Exception as exc:
                    # Detect non-structured control-flow usage.
                    from interpreter import JumpSignal

                    if isinstance(exc, JumpSignal):
                        state.goto_seen_in_frame = True
                        target = exc.target
                        key = (target.type, target.value)
                        if key not in gotopoints:
                            from interpreter import ASMRuntimeError

                            raise ASMRuntimeError(
                                f"GOTO to undefined gotopoint '{target.value}'",
                                location=statement.location,
                                rewrite_rule="GOTO",
                            )
                        i = gotopoints[key]
                        self._emit_event("after_statement", self, statement, env)
                        continue
                    raise

                self._emit_event("after_statement", self, statement, env)
                i += 1
        finally:
            state.block_stack.pop()

    def _execute_while_wrapped(self, statement, env):
        state = _get_state(self)
        state.loop_depth += 1
        try:
            return _orig_execute_while(self, statement, env)
        finally:
            state.loop_depth -= 1

    def _execute_for_wrapped(self, statement, env):
        state = _get_state(self)
        state.loop_depth += 1
        try:
            return _orig_execute_for(self, statement, env)
        finally:
            state.loop_depth -= 1

    Interpreter._execute_block = _execute_block_wrapped  # type: ignore[assignment]
    Interpreter._execute_while = _execute_while_wrapped  # type: ignore[assignment]
    Interpreter._execute_for = _execute_for_wrapped  # type: ignore[assignment]


# ---- Future-reference analysis ----


class _BlockAnalysis:
    __slots__ = (
        "statements_ref",
        "n",
        "suffix_refs",
        "suffix_prefixes",
        "suffix_calls",
        "stmt_triggers",
    )

    def __init__(
        self,
        *,
        statements_ref: List[object],
        suffix_refs: List[FrozenSet[str]],
        suffix_prefixes: List[FrozenSet[str]],
        suffix_calls: List[FrozenSet[str]],
        stmt_triggers: List[bool],
    ) -> None:
        self.statements_ref = statements_ref
        self.n = len(statements_ref)
        self.suffix_refs = suffix_refs
        self.suffix_prefixes = suffix_prefixes
        self.suffix_calls = suffix_calls
        self.stmt_triggers = stmt_triggers


def _collect_calls(node: object) -> Set[str]:
    """Collect call names within a node."""
    from parser import CallExpression

    out: Set[str] = set()
    stack: List[object] = [node]
    while stack:
        cur = stack.pop()
        if isinstance(cur, CallExpression):
            out.add(cur.name)
        stack.extend(_iter_nodes(cur))
    return out


def _statement_requirements(statement: object) -> tuple[Set[str], Set[str], Set[str], bool]:
    """Return (refs, prefixes, calls, triggers_gc) for a single statement."""
    from parser import Assignment

    refs = _collect_identifiers(statement)
    prefixes = _collect_dotted_prefixes(statement)
    calls = _collect_calls(statement)

    defines_symbol = False
    if isinstance(statement, Assignment):
        # Untyped assignment requires that the target already exists.
        if getattr(statement, "declared_type", None) is None:
            refs.add(statement.target)
        # Typed assignment can introduce a new symbol.
        else:
            defines_symbol = True

    triggers = defines_symbol or bool(refs) or bool(prefixes)
    return refs, prefixes, calls, triggers


def _get_block_analysis(interpreter, statements_list: List[object]) -> _BlockAnalysis:
    """Compute (and cache) per-block suffix reference info for fast lookups."""
    state = _get_state(interpreter)
    key = id(statements_list)
    cached = state.block_analysis_cache.get(key)
    if cached is not None and cached.statements_ref is statements_list:
        return cached

    n = len(statements_list)
    stmt_refs: List[Set[str]] = []
    stmt_prefixes: List[Set[str]] = []
    stmt_calls: List[Set[str]] = []
    stmt_triggers: List[bool] = []
    for stmt in statements_list:
        refs, prefixes, calls, triggers = _statement_requirements(stmt)
        stmt_refs.append(refs)
        stmt_prefixes.append(prefixes)
        stmt_calls.append(calls)
        stmt_triggers.append(triggers)

    # Suffix unions: position i includes statements i..end.
    suffix_refs: List[FrozenSet[str]] = [frozenset() for _ in range(n + 1)]
    suffix_prefixes: List[FrozenSet[str]] = [frozenset() for _ in range(n + 1)]
    suffix_calls: List[FrozenSet[str]] = [frozenset() for _ in range(n + 1)]
    running_refs: Set[str] = set()
    running_prefixes: Set[str] = set()
    running_calls: Set[str] = set()
    for i in range(n - 1, -1, -1):
        running_refs |= stmt_refs[i]
        running_prefixes |= stmt_prefixes[i]
        running_calls |= stmt_calls[i]
        suffix_refs[i] = frozenset(running_refs)
        suffix_prefixes[i] = frozenset(running_prefixes)
        suffix_calls[i] = frozenset(running_calls)

    analysis = _BlockAnalysis(
        statements_ref=statements_list,
        suffix_refs=suffix_refs,
        suffix_prefixes=suffix_prefixes,
        suffix_calls=suffix_calls,
        stmt_triggers=stmt_triggers,
    )
    state.block_analysis_cache[key] = analysis
    return analysis


def _iter_nodes(node: object) -> Iterable[object]:
    """Yield child nodes for the subset of parser AST we care about."""
    from parser import (
        Assignment,
        BreakStatement,
        CallArgument,
        CallExpression,
        ContinueStatement,
        ExpressionStatement,
        ForStatement,
        FuncDef,
        GotoStatement,
        GotopointStatement,
        Identifier,
        IfBranch,
        IfStatement,
        IndexExpression,
        Literal,
        Param,
        PopStatement,
        ReturnStatement,
        TensorLiteral,
        TensorSetStatement,
        WhileStatement,
    )

    if isinstance(node, Assignment):
        # Note: assignment.target handled separately (typed vs untyped).
        yield node.expression
        return
    if isinstance(node, TensorSetStatement):
        yield node.target
        yield node.value
        return
    if isinstance(node, ExpressionStatement):
        yield node.expression
        return
    if isinstance(node, IfStatement):
        yield node.condition
        yield node.then_block
        for b in node.elifs:
            yield b
        if node.else_block is not None:
            yield node.else_block
        return
    if isinstance(node, IfBranch):
        yield node.condition
        yield node.block
        return
    if isinstance(node, WhileStatement):
        yield node.condition
        yield node.block
        return
    if isinstance(node, ForStatement):
        yield node.target_expr
        yield node.block
        return
    if isinstance(node, FuncDef):
        for p in node.params:
            yield p
        yield node.body
        return
    if isinstance(node, Param):
        if node.default is not None:
            yield node.default
        return
    if isinstance(node, ReturnStatement):
        if node.expression is not None:
            yield node.expression
        return
    if isinstance(node, PopStatement):
        yield node.expression
        return
    if isinstance(node, BreakStatement):
        yield node.expression
        return
    if isinstance(node, GotoStatement):
        yield node.expression
        return
    if isinstance(node, GotopointStatement):
        yield node.expression
        return
    if isinstance(node, ContinueStatement):
        return

    # Expressions
    if isinstance(node, Identifier):
        return
    if isinstance(node, Literal):
        return
    if isinstance(node, CallExpression):
        for arg in node.args:
            yield arg
        return
    if isinstance(node, CallArgument):
        yield node.expression
        return
    if isinstance(node, TensorLiteral):
        for item in node.items:
            yield item
        return
    if isinstance(node, IndexExpression):
        yield node.base
        for idx in node.indices:
            yield idx
        return

    # Blocks
    if hasattr(node, "statements"):
        stmts = getattr(node, "statements")
        if isinstance(stmts, list):
            for s in stmts:
                yield s
        return


def _collect_identifiers(node: object) -> Set[str]:
    """Collect identifier *reads* within a node."""
    from parser import Identifier

    out: Set[str] = set()
    stack: List[object] = [node]
    while stack:
        cur = stack.pop()
        if isinstance(cur, Identifier):
            out.add(cur.name)
            continue
        stack.extend(_iter_nodes(cur))
    return out


def _collect_dotted_prefixes(node: object) -> Set[str]:
    """Collect module prefixes referenced via dotted names.

    Examples:
    - prng.PRNG_NEXT(...) -> "prng"
    - prng.MASK32 -> "prng"

    This is intentionally conservative: if any future code refers to module X via
    X.something, we treat the entire X.* namespace as potentially needed.
    """
    from parser import CallExpression, Identifier

    out: Set[str] = set()
    stack: List[object] = [node]
    while stack:
        cur = stack.pop()
        name: Optional[str] = None
        if isinstance(cur, Identifier):
            name = cur.name
        elif isinstance(cur, CallExpression):
            name = cur.name
        if name and "." in name:
            prefix = name.split(".", 1)[0]
            if prefix:
                out.add(prefix)
        stack.extend(_iter_nodes(cur))
    return out


def _collect_closure_function_refs(interpreter, env) -> tuple[Set[str], Set[str]]:
    """Conservatively treat all functions closing over `env` as future code.

    This is critical for IMPORT'd modules: module top-level executes once, but
    its globals are later referenced by exported functions via their closure.
    """
    state = _get_state(interpreter)
    cache_key = (id(env), state.function_epoch)
    cached = state.closure_refs_cache.get(cache_key)
    if cached is not None:
        return cached

    refs: Set[str] = set()
    prefixes: Set[str] = set()
    fn_table = getattr(interpreter, "functions", {})
    for fn in fn_table.values():
        try:
            closure = getattr(fn, "closure", None)
        except Exception:
            closure = None
        if closure is not env:
            continue
        body = getattr(fn, "body", None)
        if body is None:
            continue
        # Use statement-level requirements so untyped assignment targets count.
        stmts = getattr(body, "statements", None)
        if isinstance(stmts, list):
            analysis = _get_block_analysis(interpreter, stmts)
            refs |= set(analysis.suffix_refs[0])
            prefixes |= set(analysis.suffix_prefixes[0])
        else:
            refs |= _collect_identifiers(body)
            prefixes |= _collect_dotted_prefixes(body)

    out = (refs, prefixes)
    state.closure_refs_cache[cache_key] = out
    return out


def _analyze_function(interpreter, fn_name: str) -> tuple[Set[str], Set[str], Set[str]]:
    """Cached analysis of a function body: (refs, prefixes, calls)."""
    state = _get_state(interpreter)
    fn = getattr(interpreter, "functions", {}).get(fn_name)
    if fn is None:
        return set(), set(), set()
    body = getattr(fn, "body", None)
    if body is None:
        return set(), set(), set()
    body_id = id(body)
    cached = state.function_analysis_cache.get(fn_name)
    if cached is not None and cached[0] == body_id:
        return cached[1], cached[2], cached[3]

    refs: Set[str] = set()
    prefixes: Set[str] = set()
    calls: Set[str] = set()
    stmts = getattr(body, "statements", None)
    if isinstance(stmts, list):
        analysis = _get_block_analysis(interpreter, stmts)
        refs |= set(analysis.suffix_refs[0])
        prefixes |= set(analysis.suffix_prefixes[0])
        calls |= set(analysis.suffix_calls[0])
    else:
        refs |= _collect_identifiers(body)
        prefixes |= _collect_dotted_prefixes(body)
        calls |= _collect_calls(body)

    state.function_analysis_cache[fn_name] = (body_id, refs, prefixes, calls)
    return refs, prefixes, calls


def _collect_future_references(interpreter, env) -> tuple[Set[str], Set[str]]:
    """Return names referenced in any remaining statements in active blocks.

    Conservative rules:
    - In loop-executed blocks, treat the *entire* loop block as future (may repeat).
    - Treat untyped assignments (declared_type is None) as a future reference to the target
      (since deleting it would make that assignment illegal later).
    - If a GOTO has been seen in the frame, return a non-empty set that prevents collection.
    """
    state = _get_state(interpreter)

    if state.goto_seen_in_frame:
        # Disable collection once non-structured control-flow is used.
        # This is conservative: backwards jumps can reintroduce earlier references.
        return {"__gc_disabled_by_goto__"}, set()

    future_refs: Set[str] = set()
    future_prefixes: Set[str] = set()
    called_names: Set[str] = set()

    for entry in state.block_stack:
        if not entry.statements:
            continue
        analysis = _get_block_analysis(interpreter, entry.statements)

        # Important: while executing *inside* a loop body, the parent block's
        # current statement is the loop statement itself. The loop condition
        # (and possibly target expression for FOR) will be evaluated again on
        # the next iteration, so any identifiers referenced by that statement
        # must be treated as live.
        try:
            from parser import ForStatement, WhileStatement

            if 0 <= entry.index < len(entry.statements):
                cur_stmt = entry.statements[entry.index]
                if isinstance(cur_stmt, (WhileStatement, ForStatement)):
                    refs, prefixes, calls, _triggers = _statement_requirements(cur_stmt)
                    future_refs |= refs
                    future_prefixes |= prefixes
                    called_names |= calls
        except Exception:
            # If the AST shape changes or imports fail, fall back to suffix-only.
            pass

        if entry.in_loop:
            pos = 0
        else:
            pos = entry.index + 1
            if pos < 0:
                pos = 0
            if pos > analysis.n:
                pos = analysis.n
        future_refs |= set(analysis.suffix_refs[pos])
        future_prefixes |= set(analysis.suffix_prefixes[pos])
        called_names |= set(analysis.suffix_calls[pos])

    # Also treat any functions that close over the current env as future code.
    # (e.g., module globals referenced later by imported functions)
    closure_refs, closure_prefixes = _collect_closure_function_refs(interpreter, env)
    future_refs |= closure_refs
    future_prefixes |= closure_prefixes

    # Transitively include references from any potentially-called function bodies.
    fn_table = getattr(interpreter, "functions", {})
    called: Set[str] = {name for name in called_names if name in fn_table}
    seen_fns: Set[str] = set()
    pending = list(called)
    while pending:
        fn_name = pending.pop()
        if fn_name in seen_fns:
            continue
        seen_fns.add(fn_name)
        refs, prefixes, inner_calls = _analyze_function(interpreter, fn_name)
        future_refs |= refs
        future_prefixes |= prefixes
        for c in inner_calls:
            if c in fn_table and c not in seen_fns:
                pending.append(c)

    return future_refs, future_prefixes


def _gc_after_statement(interpreter, statement, env) -> None:
    # Never GC in the REPL.
    if _is_repl_interpreter(interpreter):
        return

    state = _get_state(interpreter)

    # FuncDefs change the closure landscape; bump epoch so closure caches refresh.
    try:
        from parser import FuncDef

        if isinstance(statement, FuncDef):
            state.function_epoch += 1
    except Exception:
        pass

    # If we don't have block context yet, we cannot prove anything.
    if not state.block_stack:
        return

    # Heuristic: only run the deletion pass after a statement that defines or
    # references identifiers (otherwise liveness cannot change).
    try:
        top = state.block_stack[-1]
        if 0 <= top.index < len(top.statements):
            analysis = _get_block_analysis(interpreter, top.statements)
            if not analysis.stmt_triggers[top.index]:
                return
    except Exception:
        # If anything about caching/indices is off, fall back to running GC.
        pass

    future_refs, future_prefixes = _collect_future_references(interpreter, env)

    # Only delete from the current env's own bindings; don't chase parents.
    # This matches the language's function-scoped environment model.
    to_delete: List[str] = []
    for name in list(getattr(env, "values", {}).keys()):
        if name in state.ignored:
            continue
        if name in future_refs:
            continue
        # If future code references module prefix (e.g. prng.*), keep the entire namespace.
        if name in future_prefixes:
            continue
        if "." in name:
            prefix = name.split(".", 1)[0]
            if prefix in future_prefixes:
                continue
        # Respect freezing rules: env.delete will raise; avoid raising from GC.
        if name in getattr(env, "frozen", set()) or name in getattr(env, "permafrozen", set()):
            continue
        to_delete.append(name)

    for name in to_delete:
        try:
            env.delete(name)
        except Exception:
            # GC must be non-fatal and cautious.
            pass


# ---- Public operator ----


def _gcignore_impl(interpreter, args, arg_nodes, env, location):
    from interpreter import ASMRuntimeError, TYPE_STR, Value
    from parser import Identifier

    if not arg_nodes:
        raise ASMRuntimeError("GCIGNORE expects 1 argument", location=location, rewrite_rule="GCIGNORE")

    name: Optional[str] = None

    node0 = arg_nodes[0]
    if isinstance(node0, Identifier):
        name = node0.name
    else:
        if not args:
            raise ASMRuntimeError("GCIGNORE expects 1 argument", location=location, rewrite_rule="GCIGNORE")
        v0 = args[0]
        if v0.type == TYPE_STR:
            name = str(v0.value)

    if not name:
        raise ASMRuntimeError(
            "GCIGNORE expects a symbol name (identifier or string)",
            location=location,
            rewrite_rule="GCIGNORE",
        )

    state = _get_state(interpreter)
    state.ignored.add(name)
    return Value(TYPE_STR, name)


def asm_lang_register(ext: ExtensionAPI) -> None:
    # Patch immediately so it works in REPL too (REPL doesn't emit program_start).
    _patch_interpreter_once()

    ext.metadata(name=ASM_LANG_EXTENSION_NAME, version="1.0.0")

    # Operator: GCIGNORE(symbol)
    ext.register_operator(
        "GCIGNORE",
        1,
        1,
        _gcignore_impl,
        doc="GCIGNORE(symbol): prevent cautious GC from deleting the symbol",
    )

    # Run GC after each statement.
    ext.on_event("after_statement", _gc_after_statement)
