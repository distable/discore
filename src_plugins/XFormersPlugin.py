import platform

from src.installer import pipargs
from src.classes.Plugin import Plugin


class XFormersPlugin(Plugin):
    def title(self):
        return "XFormers"

    def describe(self):
        return "Handle XFormers installation for other plugins."

    def install(self, args):