import argparse, json, secrets, sys, tempfile
from pathlib import Path


def split_cells(text):
    lines = text.splitlines(keepends=True)
    if not any(line.startswith("# %%") for line in lines): return [("code", lines)]
    cells, start, kind, found = [], 0, "code", False
    for i, line in enumerate(lines):
        if line.startswith("# %%"):
            if found or start: cells.append((kind, lines[start:i]))
            kind = "markdown" if line.startswith("# %% [markdown]") else "code"
            start, found = i + 1, True
    cells.append((kind, lines[start:]))
    return cells


def cell(kind, source):
    if kind == "markdown":
        source = [line[2:] if line.startswith("# ") else line for line in source]
    result = {"cell_type": kind, "id": secrets.token_hex(4), "metadata": {}, "source": source}
    if kind == "code": result.update(execution_count=None, outputs=[])
    return result

def convert(input_path, output_path, packages=()):
    # CRLF 를 그대로 담으면 노트북 소스에 캐리지리턴이 박힌다.
    # 윈도우에서 만든 .py 를 리눅스 서버에서 실행하므로 여기서 LF 로 통일한다.
    text = Path(input_path).read_text(encoding='utf-8')
    text = text.replace(chr(13) + chr(10), chr(10)).replace(chr(13), chr(10))
    cells = split_cells(text)
    notebook_cells = [cell(kind, source) for kind, source in cells]
    if packages: notebook_cells.insert(0, cell("code", [f"!pip install {' '.join(packages)}"]))
    notebook = {"cells": notebook_cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.11.11"}}, "nbformat": 4, "nbformat_minor": 5}
    with Path(output_path).open("w", encoding="utf-8", newline="\n") as f: json.dump(notebook, f, ensure_ascii=False, indent=2)
    check = json.loads(Path(output_path).read_text(encoding="utf-8"))
    assert check["cells"]
    print(f"wrote {output_path} ({len(check['cells'])} cells)")


def demo():
    with tempfile.TemporaryDirectory() as directory:
        input_path, output_path = Path(directory) / "in.py", Path(directory) / "out.ipynb"
        input_path.write_text("# %%\nprint('x')\n# %% [markdown]\n# hello\n# world", encoding="utf-8")
        convert(input_path, output_path)
        notebook = json.loads(output_path.read_text(encoding="utf-8"))
        assert len(notebook["cells"]) == 2
        assert notebook["cells"][1]["cell_type"] == "markdown"
        assert notebook["cells"][1]["source"] == ["hello\n", "world"]
        convert(input_path, output_path, ["numpy"])
        notebook = json.loads(output_path.read_text(encoding="utf-8"))
        assert len(notebook["cells"]) == 3
        assert notebook["cells"][0]["source"] == ["!pip install numpy"]


if __name__ == "__main__":
    if not sys.argv[1:]:
        demo()
    else:
        parser = argparse.ArgumentParser()
        parser.add_argument("input")
        parser.add_argument("output")
        parser.add_argument("--pip", nargs="+", default=[])
        args = parser.parse_args()
        convert(args.input, args.output, args.pip)
