# DEFINITIONS
#
# pid: a plugin ID, which can be either in the full form 'user/repository' or just 'repository'
# ------------------------------------------------------------

import importlib
import os
import threading
import time
import traceback
from pathlib import Path

from yachalk.ansi import Color, wrap_ansi_16

import userconf
from jargs import args
from src.classes import paths
from src.classes.logs import logplugin, logplugin_err
from src.classes.paths import short_pid
from src.classes.Plugin import Plugin
from src.lib.printlib import print_bp, print
from src.lib.printlib import trace
from src import installer

# STATE
# ----------------------------------------

plugin_dirs = []  # Plugin infos
alls = []  # Loaded plugins
num_loading = 0
loadings = []

def download_git_urls(urls: list[str], log=False):
    """
    Download plugins from GitHub into paths.plugindir
    """
    for pid in urls:
        url = pid
        if '/' in url:
            if 'http' not in pid and "github.com" not in pid:
                url = f'https://{Path("github.com/") / pid}'

            installer.gitclone(url, into_dir=paths.code_plugins)
            if log:
                logplugin(" -", url)


plugfun_img = dict(_='plugfun', type='img')

def plugfun_self(plugname):
    return dict(_='plugfun', type='self', plugname=plugname)

def plugfun_redirect(plugname, funcname):
    return dict(_='plugfun', type='redirect', plugname=plugname, funcname=funcname)

def plugfun(default_return=None):
    """
    Decorates the function to return a default value if the renderer is in dev mode.
    Later we may also support requests to an external server to get the value.
    Args:
        default_return:

    Returns:

    """

    def decorator(function):
        def wrapped_call(*kargs, **kwargs):
            from src import renderer
            if renderer.enable_dev:
                def get_retval(v):
                    if isinstance(v, dict) and v.get('_') == 'plugfun':
                        if v['type'] == 'img':
                            return renderer.rv.img
                        elif v['type'] == 'self':
                            return get(v['plugname'])
                        elif v['type'] == 'redirect':
                            return getattr(get(v['plugname']), v['funcname']).__call__(*kargs, **kwargs)
                    return v

                if isinstance(default_return, dict):
                    return get_retval(default_return)
                elif isinstance(default_return, list):
                    return [get_retval(v) for v in default_return]
                elif isinstance(default_return, tuple):
                    return tuple([get_retval(v) for v in default_return])
                elif callable(default_return):
                    return get_retval(default_return(*kargs, **kwargs))
                else:
                    return default_return

            return function(*kargs, **kwargs)

        return wrapped_call

    return decorator

def __call__(plugin):
    """
    Allows plugins.get_plug(query) like this plugins(query) instead
    """
    return get(plugin)

def get(query, instantiate=True, loading=True, installing=False):
    """
    Get a plugin instance by pid
    """
    if isinstance(query, Plugin):
        return query
    elif isinstance(query, str):
        # pid search
        for plugin in alls:
            pid = short_pid(query)
            if plugin.id.startswith(pid):
                return plugin

        if instantiate:
            for p in iter_plugins(paths.plugins):
                if p.stem == query:
                    plug = instantiate_plugin_at(p, installing)
                    if loading:
                        load(plug)
                    return plug

    return None


def instantiate_plugin_at(path: Path, with_install=False):
    """
    Create the plugin, which is a python package/directory.
    Special files are expected:
        - __init__.py: the main plugin file
        - __install__.py: the install script
        - __uninstall__.py: the uninstall script
        - __conf__.py: the configuration options for the plugin
    """
    import inspect

    with_install = with_install or args.install or userconf.install

    if not path.exists():
        return

    # Get the short pid of the plugin
    pid = paths.short_pid(path.stem)

    # Check if it's already loaded.
    matches = [p for p in alls if p.id == pid]
    if len(matches) > 0:
        print(f'Plugin {pid} is already loaded.')
        return matches[0]

    print(f"Instantiating plugin {pid} ...")

    try:
        plugin_dirs.append(path)

        # Install requirements
        # ----------------------------------------
        reqpath = (paths.code_plugins / pid / 'requirements.txt')
        if with_install and reqpath.exists():
            with trace(f'src_plugins.{path.stem}.requirements.txt'):
                print(f'Installing requirements for {pid}...')
                installer.pipreqs(reqpath)

        # Import __install__ -
        # Note: __install_ is still called even if we are not
        #       installing so that we can still declare our installations
        # ----------------------------------------
        installer.skip_installations = not with_install
        installer.default_basedir = paths.plug_repos / pid
        try:
            with trace(f'src_plugins.{path.stem}.__install__'):
                importlib.import_module(f'{paths.src_plugins_name}.{path.stem}.__install__')
        except:
            pass

        installer.default_basedir = None

        # Unpack user_conf into __conf__ (timed)
        # ----------------------------------------
        # try:
        #     with trace(f'src_plugins.{path.stem}.__conf__'):
        #         confmod = importlib.import_module(f'src_plugins.{path.stem}.__conf__')
        #         for k, v in userconf.plugins[pid].opt.items():
        #             setattr(confmod, k, v)
        # except:
        #     pass

        # NOTE:
        # We allow any github repo to be used as a discore plugin, they don't necessarily need to implement plugjobs
        if any([is_plugin_py(f, pid) for f in os.listdir(path)]):
            classtype = None

            with trace(f'src_plugins.{path.stem}.find'):
                for f in path.iterdir():
                    if is_plugin_py(f, pid):
                        with trace(f'src_plugins.{path.stem}.{f.stem}'):
                            mod = importlib.import_module(f'{paths.src_plugins_name}.{path.stem}.{f.stem}')
                            for name, member in inspect.getmembers(mod):
                                if inspect.isclass(member) and issubclass(member, Plugin) and not member == Plugin:
                                    classtype = member

            if classtype is None:
                logplugin_err(f'No plugin class found in {path}')
                return

            # Instantiate the plugin using __new__
            with trace(f'src_plugins.{path.stem}.instantiate'):
                plugin = classtype(dirpath=path)
                alls.append(plugin)
                plugin.init()

            # create directories
            plugin.res().mkdir(parents=True, exist_ok=True)
            plugin.logs().mkdir(parents=True, exist_ok=True)

            return plugin
        else:
            logplugin_err(f'No plugin class found in {path}')
        # Hence we will now begin with real discore plugin instantiation

    except Exception as e:
        logplugin_err(f"Couldn't load plugin {path.name}:")
        # mprint the exception e and its full stacktrace
        excmsg = ''.join(traceback.format_exception(None, e, e.__traceback__))
        logplugin_err(excmsg)
        plugin_dirs.remove(path)
def is_plugin_py(f, pid):
    f = Path(f)
    if f.suffix != '.py': return

    return 'plugin' in f.name.lower() \
        or f.with_suffix('').stem.lower() == pid


def instantiate_plugins_by_pids(urls: list[str], install=True):
    """
    Load plugins from a list of URLs
    """
    for pid in urls:
        instantiate_plugin_by_pid(pid, install)

def instantiate_plugin_by_pid(pid, install=True):
    if not instantiate_plugin_at(paths.plugins / short_pid(pid), install):
        for suffix in paths.plugin_suffixes:
            instantiate_plugin_at(paths.plugins / (short_pid(pid) + suffix), install)


def instantiate_plugins_in(loaddir: Path, log=False, install=True):
    """
    Load all plugin directories inside loaddir.
    """
    if not loaddir.exists():
        return

    # Read the modules from the plugin directory
    for p in iter_plugins(loaddir):
        instantiate_plugin_at(p, install)

    if log:
        logplugin(f"Instantiated {len(alls)} plugins:")
        for plugin in alls:
            print_bp(f"{plugin.id} ({plugin._dir})")

def iter_plugins(loaddir):
    for p in loaddir.iterdir():
        if p.stem.startswith('.'):
            continue

        if p.is_dir() and not p.stem.startswith('__'):
            yield p


def mod2dic(module):
    return {k: getattr(module, k) for k in dir(module) if not k.startswith('_')}

def load(plug):
    def on_before(plug):
        loadings.append(plug)

    def on_after(plug):
        plug.loaded = True
        loadings.remove(plug)

    # Load single plug
    if plug.loaded:
        logplugin_err(f"Plugin {plug.id} is already loaded!")
        return

    on_before(plug)
    invoke(plug, "load")
    on_after(plug)


def broadcast(name,
              msg=None,
              threaded=False,
              on_before=None,
              on_after=None,
              filter=None,
              *args, **kwargs):
    """
    Dispatch a function call to all plugins.
    """

    def _invoke(plug):
        global num_loading
        num_loading += 1
        if on_before: on_before(plug)
        invoke(plug, name, None, False, None, *args, **kwargs) or ret
        if on_after: on_after(plug)
        num_loading -= 1

    ret = None
    for plugin in alls:
        if filter and not filter(plugin):
            continue

        plug = get(plugin)
        # if msg and plug:
        #     logplugin(" -", msg.format(id=plug.id))

        if threaded:
            threading.Thread(target=_invoke, args=(plug,)).start()
        else:
            print(wrap_ansi_16(Color.gray.on), end="")
            _invoke(plug)
            print(wrap_ansi_16(Color.gray.off), end="")


def invoke(plugin, function, default=None, error=False, msg=None, *args, **kwargs):
    """
    Invoke a plugin
    """
    try:
        plug = get(plugin)
        if not plug:
            if error:
                logplugin_err(f"Plugin '{plugin}' not found")
            return default

        attr = getattr(plug, function, None)
        if not attr:
            if error:
                logplugin_err(f"Plugin {plugin} has no attribute {function}")

            return default

        if msg:
            plugin(msg.formagreyt(id=plug.id))

        return attr(*args, **kwargs)
    except Exception:
        logplugin_err(f"Error calling: {plugin}.{function}")
        logplugin_err(traceback.format_exc())


def wait_loading():
    """
    Wait for all plugins to finish loading.
    """
    while num_loading > 0:
        time.sleep(0.1)
