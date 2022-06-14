# -*- coding: utf-8 -*-
import shutil
import os
import nbformat

NB_VERSION = 4
TOP_LEVEL = "## "
SECTION = "####"
SLIDE_HEADER = """---\ntype: slides\n---\n\n"""
TEST_TEMPLATE = "from wasabi import msg\nfrom black import format_str, FileMode\n\nfile_mode = FileMode()\n\ndef blacken(code):\n    try:\n        return format_str(code, mode=file_mode)\n    except:\n        return code\n\n__msg__ = msg\n__solution__ = blacken(\"\"\"${solution}\"\"\")\n\n${solution}\n\n${test}\n\ntry:\n    test()\nexcept AssertionError as e:\n    __msg__.fail(e)"

base_path = os.path.abspath(".")
static_path = os.path.join(base_path, "static")
notebook_path = os.path.join(base_path, "notebooks")
chapters_path = os.path.join(base_path, "chapters", "en")
slides_path = os.path.join(chapters_path, "slides")
base_exercise_path = os.path.join(base_path, "exercises")
exercise_path = os.path.join(base_exercise_path, "en")


def create(path):
    """Create a path if it doesn't exist."""
    if not os.path.exists(path):
        os.mkdir(path)


def create_chapter_folders():
    """Create all chapter- and slide-specific folders."""
    chapters = os.path.join(base_path, "chapters")
    create(chapters)
    create(chapters_path)
    create(slides_path)


def create_exercise_folder():
    """Create all exercise folder."""
    create(base_exercise_path)
    create(exercise_path)


def create_slide_folder(module):
    """Create folders for each module's slides."""
    slides_module = os.path.join(slides_path, module)
    create(slides_module)


def copy_img_sources(module, module_name):
    """Copy all image source files to the 'static' folder, so that Gatsby can reference them."""
    create(os.path.join(static_path, module_name))
    module_img_folder = os.path.join(static_path, module_name, "img")
    create(module_img_folder)
    src = os.path.join(module, "img")
    if os.path.exists(src):
        shutil.copytree(src, module_img_folder, dirs_exist_ok=True)


def get_modules():
    """Get all notebook modules, i.e. those of the form 'notebooks/0x_foo_bar' """
    modules = [f.path for f in os.scandir(notebook_path) if f.is_dir() if "/0" in f.path]
    modules.sort()
    return modules


def get_notebook(nb_source):
    """Return a notebook source via nbformat."""
    with open(nb_source) as f:
        nb = nbformat.read(f, NB_VERSION)
    return nb


def parse_front_matter(module, module_count, num_modules):
    """Get a module's YAML front-matter."""
    with open(os.path.join(module, "README.md"), "r") as rm:
        lines = rm.readlines()
    title = lines[0].strip("# ").strip("\n")
    non_empty = [l.strip("\n") for l in lines[1:] if l]
    description = "".join(non_empty)

    prev = "null" if module_count == 0 else f"/module{module_count - 1}"
    next = "null" if module_count == num_modules - 1 else f"/module{module_count + 1}"

    front_matter = f"""---
title: "{title}"
description: \n  "{description}"
prev: {prev}
next: {next}
type: chapter
id: {module_count}
---\n\n"""

    return front_matter


def slides_and_ex(cells):
    """Split cells by "##" elements into slides and exercise groups.
    """
    # top-level slides
    top_cells = [c for c in cells if c['source'].startswith(TOP_LEVEL)]
    has_exercises = len(top_cells) >= 2

    slide_cells = cells
    ex_cells = []
    indices = [cells.index(cell) for cell in top_cells] + [10000]  # dummy index

    if has_exercises:
        slide_cells = cells[:indices[1]]
        for i in range(1, len(indices) - 1):
            ex_cells.append(cells[indices[i]:indices[i+1]])

    return slide_cells, ex_cells


def split_into_sections(cells):
    """By convention we split notebooks by sections to group slides and exercises."""
    top_cells = [c for c in cells if c['source'].startswith(SECTION)]
    indices = [cells.index(cell) for cell in top_cells] + [10000]  # dummy index
    section_cells = [cells[:indices[0]]]

    for i in range(len(indices) - 1):
        section_cells.append(cells[indices[i]:indices[i+1]])

    return section_cells


def get_slide_main(slide_cells, module_name, slide_file_name, exercise_count):
    """Get the 'exercise' tag for each slide deck in a module,
    which consists of slides and an optional video."""
    title_cell = slide_cells[0]["source"]
    if not "<!-- video" in title_cell:
        slide_title = title_cell.strip('## ')
        slide_main = f"""<exercise id="{exercise_count}" title="{slide_title}" type="slides">"""
        shot_data = ""
    else:
        slide_title = title_cell.split("\n")[0].strip('## ')
        slide_main = f"""<exercise id="{exercise_count}" title="{slide_title}" type="slides, video">"""
        shot_data = title_cell.split("\n")[1].replace("<!-- video", "").replace(" -->", "")

    slide_main += f"""<slides source="{module_name}/{slide_file_name}" {shot_data}>\n</slides>\n</exercise>\n\n"""
    return slide_main


def create_slide_file(slide_file_name, slide_cells):
    """Create the reveal.js slide decks embedded in modules."""
    module_name = slide_file_name.split("_")[0]
    slide_file = os.path.join(slides_path, module_name, slide_file_name + ".md")

    split_slide_cells = split_into_sections(slide_cells)

    with open(slide_file, mode="w", encoding="utf-8") as slide:
        slide.write(SLIDE_HEADER)

        for j, slide_cell_group in enumerate(split_slide_cells):

            notes = None
            last_cell = slide_cell_group[-1]["source"]
            if "Notes:" in last_cell:
                notes = last_cell.split("Notes:")[-1]
                cell = last_cell.split("Notes:")[0]
                slide_cell_group[-1]["source"] = cell

            for i, cell in enumerate(slide_cell_group):
                source = cell["source"]

                # Make single header lines big and centered.
                if len(source.split('\n')) == 1:
                    if "####" not in source:
                        source = source.replace("## ", "# ")
                    if i == len(slide_cell_group) - 1:
                        source = source.replace("#### ", "# ")

                # Move L4 to L2 headers
                source = source.replace("#### ", "## ")

                source = parse_source_code(cell, source, module_name)

                source = replace_images(source, module_name)
                slide.write(f"""{source}\n\n""")

            if notes:
                slide.write(f"Notes: {notes.lstrip()}\n\n---\n\n")
            else:
                if j < len(split_slide_cells) - 1:
                    slide.write("Notes:\n\n<br>\n\n---\n\n")


def strip_ansi_codes(s):
    """Remove ANSI color codes from a string.
    """
    import re
    return re.sub('\033\\[([0-9]+)(;[0-9]+)*m', '', s)


def parse_source_code(cell, source, module_name):
    if cell["cell_type"] == "code":
        source = f"""```python\n{source}\n```\n\n"""

        if cell["outputs"]:
            for output in cell["outputs"]:
                # Normal text/print output
                if output["output_type"] == "stream":
                    # ANSI character coloring is not supported in HTML
                    # (could be translated to CSS down the line).
                    out = strip_ansi_codes(output["text"])
                    source += f"""```out\n{out.lstrip()}\n\n```"""
                # Python return values
                elif output["output_type"] == "execute_result":
                    out = output["data"]["text/plain"]
                    source += f"""```out\n{out.lstrip()}\n\n```"""
                # Plotted images, no matter what plotting library
                # TODO we assume png here, but this could be generalized
                elif output["output_type"] == "display_data":
                    data = output["data"]
                    import base64

                    if "image/png" in data.keys():
                        img_data = base64.b64decode(data["image/png"])
                        alt_text = data["text/plain"]

                        import random
                        img_name = f"out_gen_{random.randint(1000, 9999)}"
                        img_path = os.path.join(static_path, module_name, f"{img_name}.png")
                        with open(img_path, 'wb') as f:
                            f.write(img_data)
                        source += f"""<center>\n<img src='/{module_name}/{img_name}.png' alt="{alt_text}"/>\n</center>\n"""

    return source


def replace_images(source, module_name):
    """Make markdown images proper HTML img tags."""
    import re
    lines = source.split("\n")
    for i, line in enumerate(lines):
        comp = re.compile(r"!\[.*]\((.+\..+)\)")
        group = comp.match(line)
        if group:
            img = group.group(1)
            lines[i] = f"""<center>\n<img src='/{module_name}/{img}' alt=""/>\n</center>\n"""

    return "\n".join(lines)


def get_exercise_title(lines, exercise_count):
    """Strip the title from an exercise cell."""
    title_content = lines[0]
    assert "## " in title_content
    ex_title = title_content.strip("## ")
    exercise_title = f"""<exercise id="{exercise_count}" title="{ex_title}">\n\n"""
    return exercise_title


def parse_exercise(ex_group, content, module_count, exercise_count):
    """Parse any exercise type."""

    lines = ex_group[0]["source"].split("\n")
    content += get_exercise_title(lines, exercise_count)
    ex_group[0]["source"] = "\n".join(lines[1:])

    choice_id = 1

    has_exercise = False
    has_solution = False
    has_test = False

    exercise_file = os.path.join(exercise_path, f"exc_{module_count}_{exercise_count}.py")
    solution_file = os.path.join(exercise_path, f"solution_{module_count}_{exercise_count}.py")
    test_file = os.path.join(exercise_path, f"test_{module_count}_{exercise_count}.py")

    for ex in ex_group:
        source = ex["source"]

        # Only parse MC question cells if they have both
        # at least a correct and an incorrect answer.
        if "- [ ]" in source and "- [x]" in source:
            found_choices = False
            lines = source.split("\n")
            content += lines[0]

            for i, line in enumerate(lines[1:]):
                if line.startswith("- ["):
                    if not found_choices:
                        content += f"""<choice id="{choice_id}">\n"""
                        found_choices = True
                        choice_id += 1
                    correct = line.startswith("- [x]")
                    text = line[6:]
                    split = text.split("|")
                    choice = split[0]

                    solution = split[1] if len(split) > 1 else ""
                    value = """correct="true" """ if correct else ""
                    opt = f"""<opt text="{choice}" {value}>\n{solution}\n</opt>\n"""
                else:
                    opt = f"{line}\n"

                content += opt

            content += "</choice>\n\n"
        elif "EXERCISE" in source:
            content += f"""<codeblock id="{module_count}_{exercise_count}">\n\n"""
            has_exercise = True
            source = source.strip("# EXERCISE")
            with open(exercise_file, "w") as exc:
                exc.write(source)
        elif "SOLUTION" in source:
            source = source.strip("# SOLUTION")
            has_solution = True
            with open(solution_file, "w") as solution:
                solution.write(source)
            content += "</codeblock>\n\n"
        elif "TEST" in source:
            source = source.strip("# TEST")
            has_test = True
            with open(test_file, "w") as test:
                test.write(source)
        else:
            content += f"{source}\n\n"

    content += "\n</exercise>\n\n"

    if has_exercise and not has_solution:
        raise Exception("Coding exercise needs at least exercise and solution")

    # If we found a coding exercise (not MC-only block), but no test, use the default.
    if has_exercise and not has_test:
        with open(test_file, "w") as test:
            test.write(TEST_TEMPLATE)

    return content
