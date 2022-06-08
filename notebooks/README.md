# How to write notebooks for this course

- The directories here contain the RL course modules.
  The course is divided into ~5 modules.
- Each module contains its own README.md with the _title_ and short _description_ of the module. 
  It is mandatory to have a markdown title (`#`) line and a body for title and description of the generated course chapter.
- Each module contains ~5 sections and each section lives in its own `.ipynb` file. 
  By convention, a section is a collection of one slide deck followed by 1-4 practice questions.
- The practice questions can be multiple choice or interactive coding exercises.
- Within one of these `.ipynb` files, 2nd level headings (`##`) are used for the slide deck title and the practice questions.
- Multiple choice questions have to have a `<!-- multiple choice -->` tag.
  Each multiple choice section needs to be a standard markdown list and have at least one selected (`- [x]`) and one unselected (`- [ ]`) list item.
- Coding exercises need to have a `<!-- coding exercise -->` tag. All exercises are written in Python.
  Each coding exercise needs to have one cell containing the exercise code and a Python comment called `# EXERCISE`.
  Every coding exercise also needs a solution cell, denoted `# SOLUTION`.
  Optionally, you can follow this up with a `# TEST` cell that's used to verify the solution.
- Within the slide decks, and practice questions, 4th level headings (`####`) are used for each slide/question. 
  If there are multiple Jupyter cells between headings, these cells are all part of the same slide.
  This is useful if, for example, a slide contains a mix of text and code.