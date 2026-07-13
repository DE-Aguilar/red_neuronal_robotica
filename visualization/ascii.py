from rich.syntax import Syntax


def horizontal_rule():
    print("\n")
    print("-" * 64)
    print("\n")


def printer(file_path):
    with open(file_path, "r") as file:
        content = file.read()
    return Syntax(content, lexer="text", background_color="default")


rocket = printer("assets/rocket.txt")
title = printer("assets/title.txt")
