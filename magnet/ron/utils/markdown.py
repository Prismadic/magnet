from rich.markdown import Markdown
from rich.console import Console

def m_print(text):
    Console().print(Markdown(str(text)))