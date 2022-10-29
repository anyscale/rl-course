import os
from utils import create_chapter_folders, create_exercise_folder, get_modules
from utils import create_slide_folder, get_notebook, slides_and_ex
from utils import create_slide_file, copy_img_sources
from utils import get_slide_main, parse_exercise, parse_front_matter
from utils import get_notebook_base_path, get_slides_path, get_exercise_path
from utils import get_chapters_path, MC, CE

# If set to True, exercise content is directly embedded into the course content, so that
# if can e.g. connect and remotely execute code on a binder server.
# If set to False, we generate separate exercise notebooks that we launch in Google
# Colab and then reference these notebook exercises in the course content.
RUNNABLE_EXERCISES = False


def generate_course_content(locale="en"):
    slides_path = get_slides_path(locale)
    exercise_path = get_exercise_path(locale)
    chapters_path = get_chapters_path(locale)

    create_chapter_folders(locale)
    create_exercise_folder(locale)

    notebook_path = get_notebook_base_path()
    notebook_path = notebook_path if locale == "en" else \
        os.path.join(notebook_path, locale)
    modules = get_modules(notebook_path)
    num_modules = len(modules)
    module_count = 0

    all_exercises_file = os.path.join(exercise_path, "exercise.py")
    with open(all_exercises_file, "w") as ex_file:
        ex_file.write("""# %% [markdown]\n\n
# Exercises and solutions for 
[Applied RL with RLlib](https://applied-rl-course.netlify.app/).\n\n
The exercises found here are immediately followed by their solutions. We encourage you 
to attempt a solution before viewing the proposed solutions. To execute the code,
please install all necessary dependencies first (keep this notebook open if you want
to run multiple exercises without having to reinstall dependencies).\n\n
# %%
! pip install ray[rllib]>=2.0.0 torch matplotlib 
! pip install gym==0.23.1 gym-toytext==0.25.0 pygame==2.1.0
# %%
!git clone https://github.com/maxpumperla/rl-course-exercises
# %%
%load rl-course-exercises/utils_01.py
%load rl-course-exercises/utils_02.py
%load rl-course-exercises/utils_03.py
%load rl-course-exercises/utils_04.py
%load rl-course-exercises/utils_05.py

%load rl-course-exercises/envs_02.py
%load rl-course-exercises/envs_03.py
%load rl-course-exercises/envs_04.py
%load rl-course-exercises/envs_05.py

%cp -r rl-course-exercises/data .
%cp -r rl-course-exercises/models .
# %%
import sys 
sys.path.insert(0, '/content/rl-course-exercises')
""")

    # For each module/chapter
    for module in modules:
        print(f">>> Writing output for module {module_count}")

        # parse the "main" markdown front-matter as module content
        content = parse_front_matter(module, module_count, num_modules)

        # Create all folders and copy static images
        module_name = f"module{module_count}"
        create_slide_folder(module_name, locale)
        copy_img_sources(module, module_name)

        # Sort all contained notebooks alphabetically
        notebooks = [
            f.path for f in os.scandir(module) if f.is_file() if ".ipynb" in f.path
        ]
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
            slide_main = get_slide_main(
                slide_cells, module_name, slide_file_name, exercise_count
            )
            content += slide_main
            exercise_count += 1

            # Write the markdown file containing the actual slide content
            create_slide_file(slide_file_name, slide_cells, slides_path)

            # Next, focus on MC and coding exercises
            for ex_group in ex_cells:
                first_source = ex_group[0]["source"]
                if MC in first_source or CE in first_source:
                    content = parse_exercise(
                        ex_group,
                        content,
                        module_count,
                        exercise_count,
                        exercise_path,
                        RUNNABLE_EXERCISES,
                        all_exercises_file
                    )
                    exercise_count += 1

            notebook_count += 1
        # Then write all the module content and move on to the next.
        with open(os.path.join(chapters_path, f"{module_name}.md"), "w") as meta:
            meta.write(content)
        module_count += 1

    import subprocess
    subprocess.run(["jupytext", all_exercises_file, "--to", "ipynb"])


if __name__ == "__main__":
    # Run "translate.py" before this script, if you haven't already done so.
    supported_locales = ["en", "de", "es", "fr"]
    for locale in supported_locales:
        generate_course_content(locale=locale)
