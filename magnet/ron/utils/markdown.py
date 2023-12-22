from rich.markdown import Markdown
from rich.console import Console

def print(text):
    Console().print(Markdown(str(text)))