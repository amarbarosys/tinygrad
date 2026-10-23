# UOp.toposort (gate=None) as a CPython extension: the same walk as tinygrad/uop/ops.py, over the same Python UOp objects.
# Measures what a Mojo rewrite engine pays when the graph stays in Python.
# build: mojo build --emit shared-lib mojo/ext/uop_topo.mojo -o .work/mojo-ext/uop_topo.so
from std.ffi import external_call
from std.os import abort
from std.python import Python, PythonObject
from std.python.bindings import PythonModuleBuilder


@export
def PyInit_uop_topo() abi("C") -> PythonObject:
    try:
        var b = PythonModuleBuilder("uop_topo")
        b.def_function[toposort]("toposort", docstring="toposort(root, call_op, enter_calls) -> dict[UOp, None]")
        b.def_function[toposort_capi]("toposort_capi", docstring="toposort_capi(id(root), id(Ops.CALL), enter_calls, id(out_dict), id('src'), id('op'))")
        return b.finalize()
    except e:
        abort(t"failed to create Python module: {e}")


def toposort(root: PythonObject, call_op: PythonObject, enter_calls: PythonObject) raises -> PythonObject:
    var cache = Python.dict()
    var enter = Bool(enter_calls)
    var stack = List[PythonObject]()
    var seen = List[Bool]()
    stack.append(root)
    seen.append(False)
    while len(stack) > 0:
        var node = stack.pop()
        var visited = seen.pop()
        if cache.__contains__(node):
            continue
        if not visited:
            stack.append(node)
            seen.append(True)
            var src = node.src
            # src_without_body: a CALL's first src is its body
            var lo = 1 if (not enter and node.op is call_op) else 0
            var i = len(src) - 1
            while i >= lo:
                stack.append(src[i])
                seen.append(False)
                i -= 1
        else:
            cache[node] = Python.none()
    return cache


# Ceiling for any engine that leaves the graph in Python: raw CPython calls on object addresses (id()), no PythonObject
# wrapper. The caller keeps every object alive; tuple items are borrowed, getattr results are released at once.
def toposort_capi(
    root: PythonObject, call_op: PythonObject, enter_calls: PythonObject, dst: PythonObject, s_src: PythonObject, s_op: PythonObject
) raises -> PythonObject:
    var call = Int(py=call_op)
    var d = Int(py=dst)
    var name_src = Int(py=s_src)
    var name_op = Int(py=s_op)
    var enter = Bool(enter_calls)
    var none = Int(py=Python.evaluate("id(None)"))
    var stack = List[Int]()
    var seen = List[Bool]()
    stack.append(Int(py=root))
    seen.append(False)
    while len(stack) > 0:
        var node = stack.pop()
        var visited = seen.pop()
        if external_call["PyDict_Contains", Int32](d, node) == 1:
            continue
        if not visited:
            stack.append(node)
            seen.append(True)
            var src = external_call["PyObject_GetAttr", Int](node, name_src)
            var lo = 0
            if not enter:
                var op = external_call["PyObject_GetAttr", Int](node, name_op)
                if op == call:
                    lo = 1
                external_call["Py_DecRef", NoneType](op)
            var i = external_call["PyTuple_Size", Int](src) - 1
            while i >= lo:
                stack.append(external_call["PyTuple_GetItem", Int](src, i))
                seen.append(False)
                i -= 1
            external_call["Py_DecRef", NoneType](src)
        else:
            _ = external_call["PyDict_SetItem", Int32](d, node, none)
    return Python.none()
