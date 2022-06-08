import os
from utils import create_chapter_folders, create_exercise_folder, get_modules, parse_front_matter
from utils import chapters_path, create_slide_folder, get_notebook, slides_and_ex
from utils import create_slide_file, copy_img_sources
from utils import get_slide_main, parse_exercise


MC = "<!-- multiple choice -->"
CE = "<!-- coding exercise -->"


create_chapter_folders()
create_exercise_folder()
modules = get_modules()
num_modules = len(modules)
module_count = 0

# For each module/chapter
for module in modules:
    print(f">>> Writing output for module {module_count}")

    # parse the "main" markdown front-matter as module content
    content = parse_front_matter(module, module_count, num_modules)

    # Create all folders and copy static images
    module_name = f"module{module_count}"
    create_slide_folder(module_name)
    copy_img_sources(module, module_name)

    # Sort all contained notebooks alphabetically
    notebooks = [f.path for f in os.scandir(module) if f.is_file() if ".ipynb" in f.path]
    notebooks.sort()

    exercise_count = 0
    notebook_count = 0
    for nb_source in notebooks:
        # Parse each notebook
        nb = get_notebook(nb_source)

        # Only parse non-empty cells that are neither hidden nor todos.
        cells = [cell for cell in nb["cells"] if cell["source"]
                 and "HIDDEN" not in cell["source"]
                 and "TODO" not in cell["source"]]

        # Split cells into slide and exercise cells
        slide_cells, ex_cells = slides_and_ex(cells)

        # Construct the import statement for slides in the main markdown file
        slide_file_name = f"{module_name}_{notebook_count}"
        slide_main = get_slide_main(slide_cells, module_name, slide_file_name, exercise_count)
        content += slide_main
        exercise_count += 1

        # Write the markdown file containing the actual slide content
        create_slide_file(slide_file_name, slide_cells)

        # Next, focus on MC and coding exercises
        for ex_group in ex_cells:
            first_source = ex_group[0]["source"]
            if MC in first_source or CE in first_source:
                content = parse_exercise(ex_group, content, module_count, exercise_count)
                exercise_count += 1

        notebook_count += 1

    # Then write all the module content and move on to the next.
    with open(os.path.join(chapters_path, f"{module_name}.md"), "w") as meta:
        meta.write(content)
    module_count += 1
