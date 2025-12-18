from __future__ import annotations

import hashlib
import importlib.util
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple


EXTENSION_API_VERSION = 1


class ASMExtensionError(Exception):
    pass


@dataclass(frozen=True)
class ExtensionMetadata:
    name: str
    version: str = "0.0.0"
    requires_api: int = EXTENSION_API_VERSION


# ---- Types ----

TypeCondition = Callable[["TypeContext", Any], int]
TypeToStr = Callable[["TypeContext", Any], str]
TypeEquals = Callable[["TypeContext", Any, Any], bool]
TypeDefault = Callable[["TypeContext"], Any]


@dataclass(frozen=True)
class TypeSpec:
    name: str
    condition_int: TypeCondition
    to_str: TypeToStr
    equals: Optional[TypeEquals] = None
    default_value: Optional[TypeDefault] = None
    printable: bool = True


@dataclass
class TypeRegistry:
    _types: Dict[str, TypeSpec] = field(default_factory=dict)
    _sealed: set[str] = field(default_factory=set)

    def seal(self, name: str) -> None:
        self._sealed.add(name)

    def register(self, spec: TypeSpec, *, seal: bool = False) -> None:
        name = spec.name
        if not name or not isinstance(name, str):
            raise ASMExtensionError("Type name must be a non-empty string")
        if name in self._types:
            raise ASMExtensionError(f"Type '{name}' is already defined")
        self._types[name] = spec
        if seal:
            self._sealed.add(name)

    def ensure_new(self, name: str) -> None:
        if name in self._sealed:
            raise ASMExtensionError(f"Type '{name}' is sealed and cannot be modified")
        if name in self._types:
            raise ASMExtensionError(f"Type '{name}' already exists")

    def has(self, name: str) -> bool:
        return name in self._types

    def get(self, name: str) -> TypeSpec:
        try:
            return self._types[name]
        except KeyError:
            raise ASMExtensionError(f"Unknown type '{name}'")

    def get_optional(self, name: str) -> Optional[TypeSpec]:
        return self._types.get(name)

    def names(self) -> set[str]:
        return set(self._types.keys())


@dataclass(frozen=True)
class StepContext:
    step_index: int
    rule: str
    location: Any  # SourceLocation | None
    extra: Optional[Dict[str, Any]]


@dataclass
class HookRegistry:
    # event -> list[(priority, handler, ext_name)]
    _events: Dict[str, List[Tuple[int, Callable[..., None], str]]] = field(default_factory=dict)
    # list[(every_n, handler, ext_name, name)]
    _step_rules: List[Tuple[int, Callable[[Any, StepContext], None], str, str]] = field(default_factory=list)
    # list[(name, runner, ext_name)]
    _repls: List[Tuple[str, Callable[["ReplContext"], int], str]] = field(default_factory=list)

    def on_event(self, event: str, handler: Callable[..., None], *, priority: int, ext_name: str) -> None:
        self._events.setdefault(event, []).append((priority, handler, ext_name))
        self._events[event].sort(key=lambda t: t[0], reverse=True)

    def emit(self, event: str, *args: Any, **kwargs: Any) -> None:
        for _priority, handler, _ext in self._events.get(event, []):
            handler(*args, **kwargs)

    def add_step_rule(self, *, name: str, every_n: int, handler: Callable[[Any, StepContext], None], ext_name: str) -> None:
        if every_n <= 0:
            raise ASMExtensionError("every_n_steps must be >= 1")
        self._step_rules.append((every_n, handler, ext_name, name))

    def after_step(self, interpreter: Any, ctx: StepContext) -> None:
        for every_n, handler, _ext, _name in self._step_rules:
            if ctx.step_index % every_n == 0:
                handler(interpreter, ctx)

    def register_repl(self, *, name: str, runner: Callable[["ReplContext"], int], ext_name: str) -> None:
        if not name:
            raise ASMExtensionError("REPL name must be non-empty")
        self._repls.append((name, runner, ext_name))

    def pick_repl(self) -> Optional[Tuple[str, Callable[["ReplContext"], int], str]]:
        if not self._repls:
            return None
        if len(self._repls) == 1:
            return self._repls[0]
        names = ", ".join(sorted({n for n, _, _ in self._repls}))
        raise ASMExtensionError(f"Multiple REPL providers registered ({names}); select one explicitly is not implemented")


@dataclass(frozen=True)
class TypeContext:
    interpreter: Any
    location: Any  # SourceLocation | None


@dataclass(frozen=True)
class ReplContext:
    verbose: bool
    services: "RuntimeServices"
    # factories / I/O
    make_interpreter: Callable[[str, str], Any]


@dataclass
class RuntimeServices:
    metadata: List[ExtensionMetadata] = field(default_factory=list)
    type_registry: TypeRegistry = field(default_factory=TypeRegistry)
    hook_registry: HookRegistry = field(default_factory=HookRegistry)
    # operators are registered directly into interpreter.builtins at attach time
    operators: List[Tuple[str, int, Optional[int], Callable[..., Any], str]] = field(default_factory=list)


class ExtensionAPI:
    def __init__(self, *, services: RuntimeServices, ext_name: str) -> None:
        self._services = services
        self._ext_name = ext_name

    # ---- metadata ----
    def metadata(self, *, name: str, version: str = "0.0.0", requires_api: int = EXTENSION_API_VERSION) -> None:
        self._services.metadata.append(ExtensionMetadata(name=name, version=version, requires_api=requires_api))

    # ---- operators ----
    def register_operator(
        self,
        name: str,
        min_args: int,
        max_args: Optional[int],
        impl: Callable[..., Any],
        *,
        doc: str = "",
    ) -> None:
        if not name:
            raise ASMExtensionError("Operator name must be non-empty")
        self._services.operators.append((name, int(min_args), None if max_args is None else int(max_args), impl, doc))

    def operator(self, name: str, min_args: int, max_args: Optional[int] = None, *, doc: str = ""):
        def deco(fn: Callable[..., Any]) -> Callable[..., Any]:
            self.register_operator(name, min_args, max_args, fn, doc=doc)
            return fn

        return deco

    # ---- types ----
    def register_type(
        self,
        name: str,
        *,
        condition_int: TypeCondition,
        to_str: TypeToStr,
        equals: Optional[TypeEquals] = None,
        default_value: Optional[TypeDefault] = None,
        printable: bool = True,
    ) -> None:
        self._services.type_registry.ensure_new(name)
        self._services.type_registry.register(
            TypeSpec(
                name=name,
                condition_int=condition_int,
                to_str=to_str,
                equals=equals,
                default_value=default_value,
                printable=printable,
            )
        )

    def type(self, name: str, *, printable: bool = True):
        def deco(fn: Callable[..., Any]) -> Callable[..., Any]:
            raise ASMExtensionError("Use register_type(...) explicitly; decorator form not supported")

        return deco

    # ---- hooks ----
    def on_event(self, event: str, handler: Optional[Callable[..., None]] = None, *, priority: int = 0):
        if handler is None:
            def deco(fn: Callable[..., None]) -> Callable[..., None]:
                self._services.hook_registry.on_event(event, fn, priority=priority, ext_name=self._ext_name)
                return fn
            return deco
        self._services.hook_registry.on_event(event, handler, priority=priority, ext_name=self._ext_name)
        return handler

    def every_n_steps(self, every_n: int, handler: Optional[Callable[[Any, StepContext], None]] = None, *, name: str = ""):
        if handler is None:
            def deco(fn: Callable[[Any, StepContext], None]) -> Callable[[Any, StepContext], None]:
                self._services.hook_registry.add_step_rule(name=name or fn.__name__, every_n=every_n, handler=fn, ext_name=self._ext_name)
                return fn
            return deco
        self._services.hook_registry.add_step_rule(name=name or handler.__name__, every_n=every_n, handler=handler, ext_name=self._ext_name)
        return handler

    # ---- repl ----
    def register_repl(self, *, name: str, runner: Callable[[ReplContext], int]) -> None:
        self._services.hook_registry.register_repl(name=name, runner=runner, ext_name=self._ext_name)


def _unique_module_name(path: str) -> str:
    base = os.path.basename(path)
    digest = hashlib.sha256(os.path.abspath(path).encode("utf-8")).hexdigest()[:12]
    safe = "".join(ch if ch.isalnum() else "_" for ch in base)
    return f"asml_ext_{safe}_{digest}"


def load_extension_module(path: str) -> Any:
    if not os.path.exists(path):
        raise ASMExtensionError(f"Extension not found: {path}")
    mod_name = _unique_module_name(path)
    spec = importlib.util.spec_from_file_location(mod_name, path)
    if spec is None or spec.loader is None:
        raise ASMExtensionError(f"Failed to load extension module: {path}")
    module = importlib.util.module_from_spec(spec)

    # Let extensions import siblings by temporarily prepending their directory.
    ext_dir = os.path.dirname(os.path.abspath(path))
    sys.path.insert(0, ext_dir)
    try:
        spec.loader.exec_module(module)  # type: ignore[union-attr]
    finally:
        if sys.path and sys.path[0] == ext_dir:
            sys.path.pop(0)
    return module


def read_asmx(pointer_file: str) -> List[str]:
    if not os.path.exists(pointer_file):
        raise ASMExtensionError(f".asmx file not found: {pointer_file}")
    base_dir = os.path.dirname(os.path.abspath(pointer_file))
    out: List[str] = []
    with open(pointer_file, "r", encoding="utf-8") as handle:
        for raw in handle.read().splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            # Allow inline comments: path # comment
            if "#" in line:
                line = line.split("#", 1)[0].strip()
                if not line:
                    continue
            if not os.path.isabs(line):
                line = os.path.abspath(os.path.join(base_dir, line))
            out.append(line)
    return out


def gather_extension_paths(paths: Sequence[str]) -> List[str]:
    expanded: List[str] = []
    for p in paths:
        if p.lower().endswith(".asmx"):
            expanded.extend(read_asmx(p))
        else:
            expanded.append(p)
    # normalize
    return [os.path.abspath(p) for p in expanded]


def build_default_services() -> RuntimeServices:
    services = RuntimeServices()
    # Reserve built-in type names so extensions cannot redefine them.
    # The interpreter will register their concrete semantics at runtime.
    services.type_registry.seal("INT")
    services.type_registry.seal("STR")
    services.type_registry.seal("TNS")
    return services


def load_runtime_services(paths: Sequence[str]) -> RuntimeServices:
    services = build_default_services()
    resolved = gather_extension_paths(paths)
    for path in resolved:
        module = load_extension_module(path)
        api_version = getattr(module, "ASM_LANG_EXTENSION_API_VERSION", EXTENSION_API_VERSION)
        if api_version != EXTENSION_API_VERSION:
            raise ASMExtensionError(
                f"Extension {path} requires API {api_version}, host supports {EXTENSION_API_VERSION}"
            )
        register = getattr(module, "asm_lang_register", None)
        if register is None or not callable(register):
            raise ASMExtensionError(f"Extension {path} must define callable asm_lang_register(ext)")
        ext_name = getattr(module, "ASM_LANG_EXTENSION_NAME", os.path.splitext(os.path.basename(path))[0])
        ext = ExtensionAPI(services=services, ext_name=str(ext_name))
        register(ext)
    return services
