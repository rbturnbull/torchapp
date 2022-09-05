from typing import List
from pybtex import PybtexEngine
from rich.console import Console

console = Console()


class Citable:
    bibtex_files = None

    def __init__(self):
        self.set_bibtex_files()

    def get_bibtex_files(self) -> List:
        return []

    def add_bibtex_file(self, file):
        if not self.bibtex_files:
            self.set_bibtext_files()

        self.bibtex_files.append(file)

    def set_bibtex_files(self):
        self.bibtex_files = self.get_bibtex_files()

    def bibtex(self) -> str:
        bibtex_strings = []
        for bibtex_file in self.bibtex_files:
            with open(bibtex_file, 'r') as f:
                bibtex_strings.append(f.read())
        return "\n".join(bibtex_strings)

    def bibliography(self, style="plain", output_backend="plaintext", **kwargs) -> str:
        engine = PybtexEngine()
        return engine.format_from_files(
            bib_files_or_filenames=self.bibtex_files, style=style, output_backend=output_backend, **kwargs
        )

    def print_bibliography(self, verbose=False, style="plain", output_backend="plaintext", **kwargs):
        bibliography = self.bibliography(style=style, output_backend=output_backend, **kwargs)
        if verbose:
            bibliography_style = "red bold"
            console.print(
                "--------------------------------------------------------------------------", style=bibliography_style
            )
            console.print(
                "Please cite these references if using this app in an academic publication:", style=bibliography_style
            )
        print(bibliography)
        if verbose:
            console.print(
                "--------------------------------------------------------------------------", style=bibliography_style
            )

    def print_bibtex(self):
        bibtex = self.bibtex()
        print(bibtex)
