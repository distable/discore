# The main core
#
# Has a global session state
# Has procedure to install plugins
# Can deploy to cloud and connect as a client to defer onto
# ----------------------------------------

from yachalk import chalk

import userconf
from src import installer, plugins
from src.classes import paths
from src.classes.logs import logcore

proxied = False
proxy = None


# class Proxy:
#     def __init__(self):
#         sio = socketio.Client()
#
#         @sio.event
#         def connect():
#             pass
#
#         @sio.event
#         def disconnect():
#             pass
#
#         self.sio = sio
#
#     def emit(self, *args, **kwargs):
#         self.sio.emit(*args, **kwargs)


def setup_annoying_logging():
    # Disable annoying message 'Some weights of the model checkpoint at openai/clip-vit-large-patch14 were not used ...'
    from transformers import logging
    logging.set_verbosity_error()
    import sys
    if not sys.warnoptions:
        import warnings
        warnings.simplefilter("ignore")

def download_plugins():
    if userconf.print_extended_init:
        print()
    logcore(chalk.green_bright("1. Downloading plugins"))
    plugins.download_git_urls([pdef.url for pdef in userconf.plugins.values()])


def create_plugins(install=True):
    if userconf.print_extended_init:
        print()
    logcore(chalk.green_bright("2. Initializing plugins"))
    # plugins.instantiate_plugins_by_pids([pdef.url for pdef in userconf.plugins.values()], install=install)
    # plugins.instantiate_plugins_in(paths.plugins, install=install)


def install_plugins():
    if userconf.print_extended_init:
        print()
    logcore(chalk.green_bright("3. Installing plugins..."))

    def before(plug):
        installer.default_basedir = paths.plug_repos / plug.short_pid

    def after(plug):
        installer.default_basedir = None

    plugins.broadcast("install", "{id}", on_before=before, on_after=after)


def unload_plugins():
    def on_after(plug):
        plug.loaded = False

    if userconf.print_extended_init:
        print()
    logcore(chalk.green_bright("Unloading plugins..."))
    plugins.broadcast("unload", "{id}", threaded=True, on_after=on_after)


def load_plugins():
    if userconf.print_extended_init:
        print()
    logcore(chalk.green_bright("3. Loading plugins..."))
    plugins.load(userconf_only=False)
