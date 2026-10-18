"""The notebooks that tell people how to run photoD name files and columns; check those names still exist.

Nothing executes the notebooks (they need the DP2 catalog), so without this a renamed column or a removed data
file leaves them quietly wrong: run_dp2.ipynb read `flags` and the patched locus for weeks after both changed.
"""

import ast
import json
from pathlib import Path

import pytest

from photod.bayes import getEstimatesMeta

REPO = Path(__file__).resolve().parents[2]
NOTEBOOKS = ["notebooks/run_dp2.ipynb"]
READER = "readAll"  # the notebooks' helper that reads columns of a fitted catalog into a dict


def codeCells(path):
    """The code cells of a notebook as parsed modules, with IPython magics and shell lines dropped."""
    cells = json.loads((REPO / path).read_text())["cells"]
    for cell in cells:
        if cell["cell_type"] != "code":
            continue
        lines = [
            line for line in "".join(cell["source"]).splitlines() if not line.lstrip().startswith(("%", "!"))
        ]
        yield ast.parse("\n".join(lines))


def strings(node):
    """The string constants of a list or tuple literal."""
    if isinstance(node, (ast.List, ast.Tuple)):
        return [e.value for e in node.elts if isinstance(e, ast.Constant) and isinstance(e.value, str)]
    return []


def columnsRead(path):
    """Every column the notebook reads from a fitted catalog: listed for READER, or a key of its result."""
    lists, readers, columns = {}, set(), set()
    for tree in codeCells(path):
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
            ):
                if strings(node.value):
                    lists[node.targets[0].id] = strings(node.value)
                call = node.value
                if isinstance(call, ast.Call) and getattr(call.func, "id", None) == READER:
                    readers.add(node.targets[0].id)
                    if len(call.args) > 1:
                        arg = call.args[1]
                        columns.update(strings(arg) or lists.get(getattr(arg, "id", None), []))
    for tree in codeCells(path):
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Subscript)
                and getattr(node.value, "id", None) in readers
                and isinstance(node.slice, ast.Constant)
                and isinstance(node.slice.value, str)
            ):
                columns.add(node.slice.value)
    return columns


@pytest.mark.parametrize("path", NOTEBOOKS)
def test_notebook_reads_only_columns_a_fit_writes(path):
    """A column renamed in the fit has to be renamed where the notebook reads it."""
    written = set(getEstimatesMeta(computeMrTrue=True).columns)
    read = columnsRead(path)
    assert read, f"{path}: found no columns read through {READER}; the test no longer sees the notebook"
    assert read <= written, f"{path} reads columns a fit does not write: {sorted(read - written)}"


@pytest.mark.parametrize("path", NOTEBOOKS)
def test_notebook_names_only_data_files_that_exist(path):
    """A file removed from data/ has to leave the notebook too."""
    named = {
        node.value
        for tree in codeCells(path)
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value.startswith("data/")
    }
    missing = sorted(name for name in named if not (REPO / name).exists())
    assert not missing, f"{path} names data files that are not in the repository: {missing}"
