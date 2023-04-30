from pathlib import Path

from . import paths


class Plugin:
    def __init__(self, dirpath: Path = None, id: str = None):
        self.jobs = []
        self.loaded = False

        # Determine our ID and directory
        if dirpath is not None and id is None:
            self._dir = Path(dirpath)
            self.id = id or Path(self._dir).stem
        elif id is not None:
            self._dir = None
            self.id = id
        else:
            raise ValueError("Either dirpath or id must be specified")

        # If the id ends with a suffix, remove it
        for suffix in paths.plugin_suffixes:
            if self.id.endswith(suffix):
                self.id = self.id[:-len(suffix)]

    @property
    def short_pid(self):
        """
        The short plugin ID.
        """
        return paths.short_pid(self.id)

    def info(self):
        """
        Get a plugin's info by ID
        """
        return dict(id=self.id,
                    jobs=self.jobs,
                    title=self.title(),
                    description=self.describe())

    def res(self, join='') -> Path:
        """
        Returns: The resource directory for this plugin
        """
        return paths.plug_res / self.id / join

    def logs(self, join='') -> Path:
        """
        Returns: The log directory for this plugin
        """
        return paths.plug_logs / self.id / join

    def repo(self, join):
        """
        Returns: The git repo dependencies directory for this plugin
        """
        return paths.plug_repos / self.id / join

    def title(self):
        """
        Display title of the plugin, for UI purposes.
        """
        raise NotImplementedError()

    def describe(self):
        """Description of the plugin, for UI purposes"""
        return ""

    def init(self):
        """
        Perform some initialization, use this instead of __init__
        """
        pass

    def load(self):
        """
        Load the models and other things into memory, such that the plugin is ready for processng.
        If enabled on startup in user_conf, this runs right after the UI is launched.
        """
        pass

    def unload(self):
        """
        Unload everything from memory.
        """
        pass

    @property
    def session(self):
        """
        Returns: The current session object
        """
        from src import renderer
        return renderer.session
