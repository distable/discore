import datetime
from pathlib import Path

import numpy as np


class AudioPlayback:
    def __init__(self):
        self.player = None
        self.paths = []
        self.markers = []

        # State
        self.iwav = None  # index of the wav to play
        self.index_playback = 0
        self.index_marker = 0
        self.start_time = 0
        self.end_time = 0
        self.start_time_wall = datetime.datetime.now()
        self.requests = []

        # Signals
        self.on_playback_start = []
        self.on_playback_stop = []

    def init(self, paths, names=None, root=None):
        from PyQt6.QtMultimedia import QAudioOutput, QMediaPlayer
        self.player = QMediaPlayer()
        self.output = QAudioOutput()
        self.output.setVolume(50)
        self.player.setAudioOutput(self.output)

        if paths == '*':
            paths = self.find_paths_in_folder(root)
        elif isinstance(paths, str):
            paths = [paths]
        elif isinstance(paths, Path):
            paths = [paths.as_posix()]

        self.paths = paths
        self.markers = []
        if paths is not None and len(paths) >= 1 and (isinstance(paths[0], str) or isinstance(paths[0], Path)):
            self.paths = []
            self.add_paths(root, paths)

    def find_paths_in_folder(self, root):
        # Get all wav, mp3, etc. files in the root
        paths = []
        for p in root.glob('*.mp3'): paths.append(p.as_posix())
        for p in root.glob('*.wav'): paths.append(p.as_posix())
        for p in root.glob('*.ogg'): paths.append(p.as_posix())
        for p in root.glob('*.flac'): paths.append(p.as_posix())
        paths = sorted(paths, key=lambda x: 'music' not in str(x))
        return paths

    def add_paths(self, root, paths):
        self.paths = []
        for pathname in paths:
            p = Path(pathname)
            if not p.is_file():
                p = root / p

            if not p.is_file():
                print(f"Could not find file {p}")
                continue

            self.paths.append(p)

    def set_wav(self, off):
        self.iwav = np.clip(self.iwav + off, 0, len(self.paths) - 1)
        self.refresh()

    def is_playing(self):
        from PyQt6.QtMultimedia import QMediaPlayer
        return self.player is not None and self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState

    def get_playback_t(self):
        elapsed = (datetime.datetime.now() - self.start_time_wall).total_seconds()
        t = self.start_time + elapsed
        return t

    def select(self, name):
        def get_wavname_index(cname):
            for i, n in enumerate(self.paths):
                if n.stem in cname:
                    return i

        if len(self.paths) == 1:
            # Auto match to the only cname
            self.iwav = get_wavname_index(self.paths[0].stem)
        else:
            # Auto-match from mouse
            self.iwav = get_wavname_index(name)

    def stop(self):
        self.requests.append(('stop',))
            # invoke_safe(self.on_playback_stop, self.start_time, self.end_time)

    def play(self, t, iwav=None, filepath=None, with_event=True):
        if t is None: return

        if len(self.paths) == 0:
            print("No audio files to play.")
            return

        if not filepath:
            iwav = iwav or self.iwav
            if iwav is None:
                iwav = 0
            iwav = np.clip(iwav, 0, len(self.paths) - 1)
            filepath = self.paths[iwav]

        self.start_time_wall = datetime.datetime.now()
        self.start_time = t
        self.requests.append(('play', filepath, t))


        # if with_event:
        #     invoke_safe(self.on_playback_start, t)

    def seek(self, t):
        if self.is_playing():
            # self.stop()
            # self.play(t, wav=self.playback_wav, with_event=False)
            self.requests.append(('seek', t))



    def refresh(self):
        if self.is_playing():
            self.stop()
            self.play(self.get_playback_t())

    def flush_requests(self):
        # self.player.setPosition(int(t * 1000))

        # self.playback_wav = filepath
        # self.player.setSource(QUrl.fromLocalFile(self.playback_wav.as_posix()))
        # self.player.setPosition(int(t * 1000))
        # self.player.play()
        from PyQt6.QtCore import QUrl

        for request in self.requests:
            rtype = request[0]
            if rtype == 'stop':
                self.end_time = self.get_playback_t()
                self.player.stop()
            elif rtype == 'play':
                filepath = request[1]
                t = request[2]
                self.player.setSource(QUrl.fromLocalFile(filepath.as_posix()))
                self.player.setPosition(int(t * 1000))
                self.player.play()
            elif rtype == 'seek':
                t = request[1]
                self.player.setPosition(int(t * 1000))
        self.requests.clear()

    # def play_marker(self):
    #     if self.markers is None or not self.markers:
    #         print("No player markers to play.")
    #         return
    #
    #     threshold = 0.1
    #     self.stop()
    #
    #     self.index_marker = np.clip(self.index_marker, 0, len(self.markers) - 1)
    #     markers = self.markers[self.index_marker]
    #
    #     self.index_playback = np.clip(self.index_playback, 0, len(markers[markers > threshold]) - 1)
    #     spent = 0
    #     for i in range(markers.shape[0]):
    #         v = markers[i]
    #         if v > threshold:
    #             spent += 1
    #             if spent == self.index_playback + 1:
    #                 print(self.index_marker, self.index_playback, i / self.fps)
    #                 self.play(i / self.fps)
    #                 break

    def get_pos(self):
        return self.player.position() / 1000
