from __future__ import annotations

from typing import Any
import tkinter as tk
import tkinter.font as tkfont

class TextLineNumbers(tk.Canvas):
    """Small line-number gutter bound to a Tk Text widget."""

    def __init__(self, master: tk.Widget, text_widget: tk.Text, **kwargs: Any) -> None:
        super().__init__(master, width=54, highlightthickness=0, **kwargs)
        self.text_widget = text_widget
        self._font = tkfont.Font(text_widget, text_widget.cget("font"))

    def redraw(self, *_args: object) -> None:
        self.delete("all")
        i = self.text_widget.index("@0,0")
        while True:
            dline = self.text_widget.dlineinfo(i)
            if dline is None:
                break
            y = dline[1]
            line_no = i.split(".", 1)[0]
            self.create_text(48, y, anchor="ne", text=line_no, font=self._font, fill="#777777")
            i = self.text_widget.index(f"{i}+1line")

