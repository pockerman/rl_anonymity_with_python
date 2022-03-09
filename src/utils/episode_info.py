"""Module episode_info. Specifies the  dataclass
EpisodeInfo that is used as the return item of on_episode() agent
function to wrap episode results. This is a helper class
to wrap the output after an episode has finished

"""

from dataclasses import dataclass, field


@dataclass(init=True, repr=True)
class EpisodeInfo(object):

    episode_itrs: int = 0
    episode_score: float = 0.0
    total_distortion: float = 0.0
    total_execution_time: float = 0.0
    info: dict = field(default_factory=dict)
