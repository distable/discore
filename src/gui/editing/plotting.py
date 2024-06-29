import matplotlib.pyplot as plt

from .utils import compute_bbox_of_points
from .jtext import JtextROI
from .jtext import Jtext
from .jcomposition import Jcomposition


class JPlotting:

    def __init__(self, objects):

        self.objects = [obj for obj in objects]
        self._bbox = None

    def add_object(self, obj):
        self.objects.append(obj)

    def compute_bbox(self):

        def _add_bbox_to_set_of_bbox(_set_of_bbox, _points):
            bbox = compute_bbox_of_points(_points)
            _set_of_bbox.append([bbox[0][0], bbox[0][1]])
            _set_of_bbox.append([bbox[1][0], bbox[1][1]])

        set_of_bbox = []
        for obj in self.objects:

            if type(obj) is Jcomposition:
                drawing_points = obj.get_raster_points()
                for points in drawing_points:
                    _add_bbox_to_set_of_bbox(set_of_bbox, points)
                continue

            drawing_points_method = getattr(obj, "get_drawing_points", None)
            if callable(drawing_points_method):
                points = obj.get_raster_points()
                _add_bbox_to_set_of_bbox(set_of_bbox, points)

        self._bbox = compute_bbox_of_points(set_of_bbox)

    def plot(self, filename, xkcd=False, size=None):

        plt.close("all")

        self.compute_bbox()
        x0, y0 = self._bbox[0]
        x1, y1 = self._bbox[1]

        width = x1 - x0
        height = y1 - y0

        if xkcd:
            plt.xkcd()

        if size is not None:
            plt.figure(figsize=size)

        for obj in self.objects:

            if type(obj) is Jcomposition:
                drawing_points = obj.get_raster_points()
                for points, o in zip(drawing_points, obj.objects):
                    color = o.color
                    self._plot(points, color)
            elif type(obj) is JtextROI:
                self._plot_text(obj)
            elif type(obj) is Jtext:
                pass
            else:
                points = obj.get_raster_points()
                self._plot(points, obj.color)

        eps_x = 3 * width / 100
        eps_y = 3 * height / 100
        plt.xlim([x0 - eps_x, x1 + eps_x])
        plt.ylim([y0 - eps_y, y1 + eps_y])

        plt.axis("off")
        plt.axes().set_aspect("equal")

        plt.tight_layout()
        plt.savefig(filename)

        # resets the xkcd style
        plt.rcdefaults()

    @staticmethod
    def _plot(points, color):
        x = [p[0] for p in points]
        y = [p[1] for p in points]
        plt.plot(x, y, color=color, linestyle="-")

    @staticmethod
    def _plot_text(text_roi):

        x = text_roi.pos().x()
        y = text_roi.pos().y()

        size = text_roi.size
        color = text_roi.color
        text = text_roi.text.textItem.toPlainText()
        transpose = text_roi.transpose

        if transpose:
            plt.text(x, y, text, color=color, va="bottom", ha="center",
                     rotation="vertical", size=size)
        else:
            plt.text(x, y, text, color=color, ha="left", va="top", size=size)
