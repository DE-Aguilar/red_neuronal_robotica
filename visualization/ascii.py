from rich.syntax import Syntax


def horizontal_line():
    return f"\n {'-' * 64}\n"


def printer(file_path):
    with open(file_path, "r") as file:
        content = file.read()
    return Syntax(content, lexer="text", background_color="default")


horizontal_line = horizontal_line()
rocket = printer("assets/rocket.txt")
title = printer("assets/title.txt")
