"""Keras tutobooks implementation.

A tutobook is a tutorial available simultaneously as a notebook,
as a Python script, and as a nicely rendered webpage.

Its source-of-truth (for manual edition and version control) is
its Python script form, but you can also create one by starting
from a notebook and converting it with the command `nb2py`.

Text cells are stored in markdown-formatted comment blocks.
the first line (starting with " * 3) may optionally contain a special
annotation, one of:

- invisible: do not render this block.
- shell: execute this block while prefixing each line with `!`.

The script form should start with a header with the following fields:
Title:
Author: (could be `Authors`: as well, and may contain markdown links)
Date created: (date in yyyy/mm/dd format)
Last modified: (date in yyyy/mm/dd format)
Description: (one-line text description)

## How to add a new code example to Keras.io

You would typically start from an existing notebook.

Save it to disk (let's say as `path_to_your_nb.ipynb`).
`cd` to the `keras-io/scripts/` directory.

Then run:

```
python tutobooks nb2py path_to_your_nb.ipynb ../examples/your_example.py
```

This will create the file `examples/your_example.py`. Open it,
fill in the headers, and generally edit it so that it looks nice.

NOTE THAT THE CONVERSION SCRIPT MAY MAKE MISTAKES IN ITS ATTEMPTS
TO SHORTEN LINES. MAKE SURE TO PROOFREAD THE GENERATED .py IN FULL.
Or alternatively, make sure to keep your lines reasonably-sized (<90 char)
to start with, so that the script won't have to shorten them.

You can then preview what it looks like when converted back again
to ipynb by running:

```
python tutobooks py2nb ../examples/your_example.py preview.ipynb
```

NOTE THAT THIS COMMAND WILL ERROR OUT IF ANY CELLS TAKES TOO LONG
TO EXECUTE. In that case, make your code lighter/faster.
Remember that examples are meant to demonstrate workflows, not
train state-of-the-art models. They should
stay very lightweight.

Open the generated `preview.ipynb` and make sure it looks like what
you expect. If not, keep editing `your_example.py` until it does.

Finally, submit a PR adding `examples/your_example.py`.
"""

import os
import re
import sys
import json
import copy
import random
import shutil
import tempfile
from pathlib import Path

TIMEOUT = 12 * 60 * 60  # 12 hours
MAX_LOC = 350


def nb_to_py(nb_path, py_path):
    f = open(nb_path)
    content = f.read()
    f.close()
    nb = json.loads(content)
    py = '"""\n'
    py += "Title: FILLME\n"
    py += "Author: FILLME\n"
    py += "Date created: FILLME\n"
    py += "Last modified: FILLME\n"
    py += "Description: FILLME\n"
    py += '"""\n'
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            # Is it a shell cell?
            if cell["source"] and cell["source"][0] and cell["source"][0][0] == "!":
                # It's a shell cell
                py += '"""shell\n'
                py += "".join(cell["source"]) + "\n"
                py += '"""\n\n'
            else:
                # It's a Python cell
                py += "".join(cell["source"]) + "\n\n"
        elif cell["cell_type"] == "markdown":
            py += '"""\n'
            py += "".join(cell["source"]) + "\n"
            py += '"""\n\n'
    # Save file
    f = open(py_path, "w")
    f.write(py)
    f.close()
    # Format file with Black
    os.system("black " + py_path)
    # Shorten lines
    py = open(py_path).read()
    try:
        py = _shorten_lines(py)
    finally:
        f = open(py_path, "w")
        f.write(py)
        f.close()


def py_to_nb(py_path, nb_path, fill_outputs=False):
    f = open(py_path)
    py = f.read()
    f.close()
    validate(py)

    header, _, py, tag = _get_next_script_element(py)
    attributes = _parse_header(header)
    cells = []
    loc = 0
    # Write first header cell
    header_cell = {
        "cell_type": "markdown",
        "source": [
            "# " + attributes["title"] + "\n",
            "\n",
            "**" + attributes["auth_field"] + ":** " + attributes["author"] + "<br>\n",
            "**Date created:** " + attributes["date_created"] + "<br>\n",
            "**Last modified:** " + attributes["last_modified"] + "<br>\n",
            "**Description:** " + attributes["description"],
        ],
        "metadata": {"colab_type": "text"},
    }
    cells.append(header_cell)
    while py:
        e, cell_type, py, tag = _get_next_script_element(py)
        lines = e.split("\n")

        if all(l == "" for l in lines):
            continue

        if lines and not lines[0]:
            lines = lines[1:]
        source = [l + "\n" for l in lines]
        # Drop last newline char
        if source and not source[-1].strip():
            source = source[:-1]
        if source:
            source[-1] = source[-1].rstrip()
        if tag == "shell":
            source = ["!" + l for l in source]
            cell_type = "code"
        if tag != "invisible" and source:
            cell = {"cell_type": cell_type, "source": source}
            if cell_type == "code":
                cell["outputs"] = []
                cell["metadata"] = {"colab_type": "code"}
                cell["execution_count"] = 0
                loc += _count_locs(source)
            else:
                cell["metadata"] = {"colab_type": "text"}
            cells.append(cell)
    notebook = {}
    for key in NB_BASE.keys():
        notebook[key] = copy.deepcopy(NB_BASE[key])
    notebook["metadata"]["colab"]["name"] = str(py_path).split("/")[-1][:-3]
    notebook["metadata"]["accelerator"] = attributes["accelerator"]
    notebook["cells"] = cells
    if loc > MAX_LOC:
        raise ValueError(
            f"Found {loc} lines of code, but expected fewer than {MAX_LOC}"
        )

    f = open(nb_path, "w")
    f.write(json.dumps(notebook, indent=1, sort_keys=True))
    f.close()
    if fill_outputs:
        print("Generating ipynb")
        parent_dir = Path(nb_path).parent
        current_files = os.listdir(parent_dir)
        try:
            os.system(
                "jupyter nbconvert --to notebook --execute --debug "
                + str(nb_path)
                + " --inplace"
                + " --ExecutePreprocessor.timeout="
                + str(TIMEOUT)
            )
        finally:
            new_files = os.listdir(parent_dir)
            for fname in new_files:
                if fname not in current_files:
                    fpath = parent_dir / fname
                    if os.path.isdir(fpath):
                        print("Removing created folder:", fname)
                        shutil.rmtree(fpath)
                    else:
                        print("Removing created file:", fname)
                        os.remove(fpath)


def nb_to_md(nb_path, md_path, img_dir, working_dir=None):
    img_exts = ("png", "jpg", "jpeg")
    # Assumes an already populated notebook.
    assert str(md_path).endswith(".md")
    current_dir = os.getcwd()
    original_img_dir = str(img_dir)
    if original_img_dir.endswith("/"):
        original_img_dir = original_img_dir[:-1]
    img_dir = os.path.abspath(img_dir)
    nb_path = os.path.abspath(nb_path)
    nb_fname = str(nb_path).split(os.path.sep)[-1]

    del_working_dir = False
    if working_dir is None:
        del_working_dir = True
        working_dir = "tmp_" + str(random.randint(1e6, 1e7))
    if not os.path.exists(working_dir):
        os.makedirs(working_dir)
    print("Using working_dir:", working_dir)

    os.chdir(working_dir)
    shutil.copyfile(nb_path, nb_fname)

    md_name = str(md_path).split("/")[-1][:-3]
    target_md = md_name + ".md"
    img_dir = Path(img_dir) / md_name
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

    os.system(
        "jupyter nbconvert --to markdown --execute --debug "
        + nb_fname
        + " --output "
        + target_md
        + " --ExecutePreprocessor.timeout="
        + str(TIMEOUT)
    )
    if os.path.exists(md_name + ".md"):
        success = True
        tmp_img_dir = md_name + "_files"
        if os.path.exists(tmp_img_dir):
            for fname in os.listdir(tmp_img_dir):
                if fname.endswith(img_exts):
                    src = Path(tmp_img_dir) / fname
                    target = Path(img_dir) / fname
                    print("copy", src, "to", target)
                    shutil.copyfile(src, target)

        os.chdir(current_dir)
        md_content = open(Path(working_dir) / (md_name + ".md")).read()
        for ext in img_exts:
            md_content = md_content.replace(
                "![" + ext + "](" + md_name + "_files",
                "![" + ext + "](" + original_img_dir + "/" + md_name,
            )
        md_content = _make_output_code_blocks(md_content)
        open(md_path, "w").write(md_content)
    else:
        success = False
        os.chdir(current_dir)

    if del_working_dir:
        shutil.rmtree(working_dir)

    if not success:
        raise RuntimeError(
            "An error was encountered when attempting to run the notebook. "
            "See logs for details."
        )


def py_to_md(py_path, nb_path, md_path, img_dir, working_dir=None):
    py_to_nb(py_path, nb_path, fill_outputs=False)
    nb_to_md(nb_path, md_path, img_dir, working_dir=working_dir)


def validate(py):
    """Validate the format of a tutobook script.

    Specifically:
        - validate headers
        - validate style with black
    """
    lines = py.split("\n")
    if not lines[0].startswith('"""'):
        raise ValueError('Missing `"""`-fenced header at top of script.')
    if not lines[1].startswith("Title: "):
        raise ValueError("Missing `Title:` field.")
    if not lines[2].startswith("Author: ") and not lines[2].startswith("Authors: "):
        raise ValueError("Missing `Author:` field.")
    if not lines[3].startswith("Date created: "):
        raise ValueError("Missing `Date created:` field.")
    if not lines[4].startswith("Last modified: "):
        raise ValueError("Missing `Last modified:` field.")
    if not lines[5].startswith("Description: "):
        raise ValueError("Missing `Description:` field.")
    if not lines[6].startswith("Accelerator: "):
        raise ValueError("Missing `Accelerator:` field.")
    description = lines[5][len("Description: ") :]
    if not description:
        raise ValueError("Missing `Description:` field content.")
    if not description[0] == description[0].upper():
        raise ValueError("Description field content must be capitalized.")
    if not description[-1] == ".":
        raise ValueError("Description field content must end with a period.")
    if len(description) > 100:
        raise ValueError("Description field content must be less than 100 chars.")
    accelerator = lines[6][len("Accelerator: ") :]
    accelerator_options = ["GPU", "TPU", "None"]
    if accelerator not in accelerator_options:
        raise ValueError(
            f"Accelerator field content must be one of: {accelerator_options}. "
            f"Received: accelerator={accelerator}"
        )
    for i, line in enumerate(lines):
        if line.startswith('"""') and line.endswith('"""') and len(line) > 3:
            raise ValueError(
                'Do not use single line `"""`-fenced comments. '
                "Encountered at line %d" % (i,)
            )
    for i, line in enumerate(lines):
        if line.endswith(" "):
            raise ValueError("Found trailing space on line %d; line: `%s`" % (i, line))
    # Validate style with black

    tmp = tempfile.gettempdir()
    fpath = os.path.join(tmp, str(random.randint(1e6, 1e7)) + ".py")
    f = open(fpath, "w")
    pre_formatting = "\n".join(lines)
    f.write(pre_formatting)
    f.close()
    os.system("black " + fpath)
    f = open(fpath)
    formatted = f.read()
    f.close()
    os.remove(fpath)
    if formatted != pre_formatting:
        raise ValueError(
            "Your python file did not follow `black` conventions. "
            "Run `black your_file.py` to autoformat it."
        )

    # Extra checks.
    if "//arxiv.org/pdf/" in py:
        raise ValueError(
            "Do not link to arXiv PDFs directly. " "Instead, link to the abstract page."
        )


def count_locs_in_file(py_path):
    f = open(py_path)
    py = f.read()
    f.close()
    _get_next_script_element(py)  # Header
    loc = 0
    while py:
        e, cell_type, py, _ = _get_next_script_element(py)
        lines = e.split("\n")
        if cell_type == "code":
            loc += _count_locs(lines)
    return loc


def _count_locs(lines):
    loc = 0
    string_open = False
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if not string_open:
            if not line.startswith('"""'):
                loc += 1
            else:
                if not line.endswith('"""'):
                    string_open = True
        else:
            if line.startswith('"""'):
                string_open = False
    return loc


def _shorten_lines(py):
    max_len = 90
    lines = []
    for line in py.split("\n"):
        if len(line) <= max_len:
            lines.append(line)
            continue
        i = 0
        while len(line) > max_len:
            line = line.lstrip()
            if " " not in line[1:]:
                lines.append(line)
                break
            else:
                short_line = line[:max_len]
                line = line[max_len:]
                if " " in short_line:
                    reversed_short_line = short_line[::-1]
                    index = reversed_short_line.find(" ") + 1
                    line = short_line[-index:] + line
                    short_line = short_line[:-index]

                lines.append(short_line.lstrip())
            i += 1
            if i > 10:
                raise
        lines.append(line.lstrip())
    return "\n".join(lines)


def _get_next_script_element(py):
    lines = py.split("\n")
    assert lines
    elines = []
    i = 0
    tag = None
    if lines[0].startswith('"""'):
        assert len(lines) >= 2
        etype = "markdown"
        if len(lines[0]) > 3:
            tag = lines[0][3:]
            if tag not in ["shell", "invisible"]:
                raise ValueError("Found unknown cell tag:", tag)
        lines = lines[1:]
    else:
        etype = "code"

    for i, line in enumerate(lines):
        if line.startswith('"""'):
            break
        else:
            elines.append(line)

    if etype == "markdown":
        py = "\n".join(lines[i + 1 :])
    else:
        py = "\n".join(lines[i:])
    e = "\n".join(elines)

    return e, etype, py, tag


def _parse_header(header):
    lines = header.split("\n")
    if len(lines) not in (6, 7):
        raise ValueError("Invalid header, it should be exactly 6 or 7 lines.")
    title = lines[0][len("Title: ") :]
    author_line = lines[1]
    if author_line.startswith("Authors"):
        author = author_line[len("Authors: ") :]
        auth_field = "Authors"
    else:
        author = author_line[len("Author: ") :]
        auth_field = "Author"
    date_created = lines[2][len("Date created: ") :]
    last_modified = lines[3][len("Last modified: ") :]
    description = lines[4][len("Description: ") :]
    accelerator = lines[5][len("Accelerator: ") :]
    return {
        "title": title,
        "author": author,
        "auth_field": auth_field,
        "date_created": date_created,
        "last_modified": last_modified,
        "description": description,
        "accelerator": accelerator,
    }


def _make_output_code_blocks(md):
    lines = md.split("\n")
    output_lines = []
    final_lines = []
    is_inside_backticks = False

    def is_output_line(line, prev_line, output_lines):
        if (
            output_lines
            and "\x08" in output_lines[-1]
            and not output_lines[-1].replace("\x08", "").strip()
        ):
            # We already started a block and the last line is just deletes, that
            # implies that the current line will be part of the block (progress
            # bar being re-written).
            return True

        if line.startswith("    ") and len(line) >= 5:
            # Non-empty indented line
            if output_lines:
                # Continuation of the output block
                return True
            if not prev_line.strip():
                # Begining of an output block
                return True
        elif not line.strip():
            # Empty line
            if output_lines:
                # Continuation of the output block
                return True
        elif line.strip()[0] in ("\x1b", "\x08"):
            # Line starts with ESC or delete character, it must be a progress
            # bar, which is often not indented.
            return True
        return False

    def flush(output_lines, final_lines):
        final_lines.append('<div class="k-default-codeblock">')
        final_lines.append("```")

        # When not None, we are in a progress bar and this is its last state.
        progress_bar = None
        # Used to dedupe empty lines. Also used when in a progress bar.
        previous_line_empty = False

        for line in output_lines:
            # Unindent.
            if line.startswith("    "):
                # Normal block is indented by 4 spaces.
                line = line[4:]
            else:
                # Progress bar and empty lines.
                line = line.strip()

            if "\x1b" in line or "\x08" in line:
                # This is a progress bar.
                if "\x1b" in line:
                    # Remove escape sequences.
                    line = re.sub(r"\x1b\[[0-9][0-9]?m", "", line)

                if "\x08" in line:
                    # Delete characters, remove everything up to the last one.
                    line = line[line.rindex("\x08") + 1 :].strip()

                if previous_line_empty and progress_bar is None:
                    # We're starting a progress bar, flush the empty line.
                    final_lines.append("")

                if progress_bar is None or line:
                    # Update latest progress bar content.
                    progress_bar = line

                previous_line_empty = not line
                # When in a progress bar, don't append.
                continue

            if progress_bar is not None and not line:
                # In a progress bar with an empty line.
                previous_line_empty = True
                # We're staying in the progress bar, don't append.
                continue

            # If we get here, we're not / no longer in a progress bar.

            if progress_bar:
                # Flush progress bar content with the last value.
                final_lines.append(progress_bar)
                progress_bar = None

            if line:
                if previous_line_empty:
                    # Flush empty line before appending non-empty line.
                    final_lines.append("")
                final_lines.append(line)
                previous_line_empty = False
            else:
                previous_line_empty = True

        if progress_bar:
            # Flush progress bar content with the last value.
            final_lines.append(progress_bar)

        final_lines.append("```")
        final_lines.append("</div>")

        if previous_line_empty:
            # If the last line in the block was empty, put it after the block.
            final_lines.append("")

    for line in lines:
        if line.startswith("```"):
            is_inside_backticks = not is_inside_backticks
            final_lines.append(line)
            continue

        if is_inside_backticks:
            final_lines.append(line)
            continue

        if final_lines and is_output_line(line, final_lines[-1], output_lines):
            output_lines.append(line)
        elif not line:
            if output_lines:
                if output_lines[-1]:
                    output_lines.append(line)
            else:
                final_lines.append(line)
        else:
            if output_lines:
                flush(output_lines, final_lines)
                output_lines = []
            final_lines.append(line)
    if output_lines:
        flush(output_lines, final_lines)
    return "\n".join(final_lines)


NB_BASE = {
    "metadata": {
        "accelerator": "GPU",
        "colab": {
            "collapsed_sections": [],
            "name": "",  # FILL ME
            "private_outputs": False,
            "provenance": [],
            "toc_visible": True,
        },
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.7.0",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd not in {"nb2py", "py2nb", "count_loc"}:
        raise ValueError(
            "Specify a command: either "
            "`nb2py source_filename.ipynb target_filename.py` or "
            "`py2nb source_filename.py target_file name.ipynb` or "
            "`count_loc source_filename.py`."
        )
    if cmd == "count_loc":
        source = sys.argv[2]
        loc = count_locs_in_file(source)
        print(f"Counted {loc} lines of code in {source}.")
    else:
        if len(sys.argv) < 4:
            raise ValueError("Specify a source filename and a target filename")
        source = sys.argv[2]
        target = sys.argv[3]

        if cmd == "py2nb":
            if not source.endswith(".py"):
                raise ValueError(
                    f"The source filename should be a Python file. Got: {source}"
                )
            if not target.endswith(".ipynb"):
                raise ValueError(
                    f"The target filename should be a notebook file. Got: {target}"
                )
            py_to_nb(source, target)
        if cmd == "nb2py":
            if not source.endswith(".ipynb"):
                raise ValueError(
                    f"The source filename should be a notebook file. Got: {source}"
                )
            if not target.endswith(".py"):
                raise ValueError(
                    f"The target filename should be a Python file. Got: {target}"
                )
            nb_to_py(source, target)
