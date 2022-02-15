"""
EpisodeInfo class. This is a helper class
to wrap the output after an episode has finished
"""


class EpisodeInfo(object):

    def __init__(self):
        self.episode_score = None
        self.total_distortion = None
        self.info = {}