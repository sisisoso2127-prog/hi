"""A minimal vector canvas that emits LaTeX `picture` primitives.

No TikZ, no matplotlib: every figure in these documents is built from filled
rectangles placed with \\put, which base LaTeX and xcolor provide.  A segment
is drawn by stepping small squares along it, so arbitrary slopes work.
"""
from math import hypot


class Canvas:
    def __init__(self, w, h, unit=1.0):
        self.w, self.h, self.unit = w, h, unit
        self.parts = []

    # ---- primitives ------------------------------------------------------
    def rule(self, x, y, w, h, color="black"):
        if w <= 0 or h <= 0:
            return
        self.parts.append(
            f"\\put({x:.2f},{y:.2f}){{\\color{{{color}}}\\rule{{{w:.2f}\\unitlength}}"
            f"{{{h:.2f}\\unitlength}}}}")

    def seg(self, x1, y1, x2, y2, color="black", width=0.7, dash=None):
        d = hypot(x2 - x1, y2 - y1)
        if d == 0:
            return self.rule(x1, y1, width, width, color)
        steps = max(2, int(d / (width * 0.55)))
        for i in range(steps + 1):
            if dash and (i // dash) % 2:
                continue
            t = i / steps
            self.rule(x1 + t * (x2 - x1) - width / 2,
                      y1 + t * (y2 - y1) - width / 2, width, width, color)

    def frame(self, x, y, w, h, color="black", width=0.7, dash=None):
        self.seg(x, y, x + w, y, color, width, dash)
        self.seg(x + w, y, x + w, y + h, color, width, dash)
        self.seg(x + w, y + h, x, y + h, color, width, dash)
        self.seg(x, y + h, x, y, color, width, dash)

    def fill(self, x, y, w, h, color):
        self.rule(x, y, w, h, color)

    def dot(self, x, y, r=2.0, color="black"):
        self.rule(x - r, y - r, 2 * r, 2 * r, color)

    def ring(self, x, y, r=2.6, color="black", width=0.7):
        """A hollow square outline.

        Hollow rather than white-filled on purpose: a ring is drawn *over* the
        point it marks, and a filled interior would erase it -- which silently
        turned an efficient point into a dominated-looking one in the first
        version of these figures.
        """
        self.rule(x - r, y - r, 2 * r, width, color)
        self.rule(x - r, y + r - width, 2 * r, width, color)
        self.rule(x - r, y - r, width, 2 * r, color)
        self.rule(x + r - width, y - r, width, 2 * r, color)

    def text(self, x, y, s, anchor="l", size="\\scriptsize"):
        box = {"l": "[l]", "r": "[r]", "c": ""}[anchor]
        self.parts.append(
            f"\\put({x:.2f},{y:.2f}){{\\makebox(0,0){box}{{{size} {s}}}}}")

    # ---- output ----------------------------------------------------------
    def tex(self):
        body = "\n".join("  " + p for p in self.parts)
        return (f"\\setlength{{\\unitlength}}{{{self.unit}pt}}%\n"
                f"\\begin{{picture}}({self.w:.0f},{self.h:.0f})\n{body}\n"
                f"\\end{{picture}}")


class Axes(Canvas):
    """A canvas with a data-to-point mapping and drawn axes."""

    def __init__(self, w, h, xlim, ylim, pad=26, unit=1.0):
        super().__init__(w, h, unit)
        self.x0, self.x1 = xlim
        self.y0, self.y1 = ylim
        self.pad = pad
        self.iw = w - pad - 8
        self.ih = h - pad - 10

    def X(self, v):
        return self.pad + (float(v) - self.x0) / (self.x1 - self.x0) * self.iw

    def Y(self, v):
        return self.pad + (float(v) - self.y0) / (self.y1 - self.y0) * self.ih

    def axes(self, xlabel, ylabel, xticks=None, yticks=None, color="black!55"):
        self.seg(self.pad - 6, self.pad, self.pad + self.iw + 6, self.pad, color, 0.7)
        self.seg(self.pad, self.pad - 6, self.pad, self.pad + self.ih + 6, color, 0.7)
        for v in (xticks or []):
            x = self.X(v)
            self.seg(x, self.pad - 2.5, x, self.pad, color, 0.6)
            self.text(x, self.pad - 8, f"${v}$", "c")
        for v in (yticks or []):
            y = self.Y(v)
            self.seg(self.pad - 2.5, y, self.pad, y, color, 0.6)
            self.text(self.pad - 6, y, f"${v}$", "r")
        self.text(self.pad + self.iw + 9, self.pad - 1, xlabel, "l")
        self.text(self.pad - 2, self.pad + self.ih + 9, ylabel, "r")

    def box(self, lo, hi, color, alpha_color=None, dash=None, label=None,
            width=0.7):
        """An axis-aligned criterion-space box; +-inf clipped to the frame."""
        xa = self.X(max(lo[0], self.x0)) if lo[0] is not None else self.pad
        xb = self.X(min(hi[0], self.x1)) if hi[0] is not None else self.pad + self.iw
        ya = self.Y(max(lo[1], self.y0)) if lo[1] is not None else self.pad
        yb = self.Y(min(hi[1], self.y1)) if hi[1] is not None else self.pad + self.ih
        if alpha_color:
            self.fill(xa, ya, max(xb - xa, 0), max(yb - ya, 0), alpha_color)
        self.frame(xa, ya, max(xb - xa, 0), max(yb - ya, 0), color, width, dash)
        if label:
            self.text((xa + xb) / 2, (ya + yb) / 2, label, "c")
