from dataclasses import dataclass
import random


@dataclass(frozen=True)
class Episode:
    classes: tuple[str, str, str]
    support_indices: tuple[int, ...]
    query_indices: tuple[int, ...]
    support_labels: tuple[int, ...]
    query_labels: tuple[int, ...]

    @property
    def indices(self):
        return self.support_indices + self.query_indices


class EpisodeSampler:
    def __init__(self, class_indices, way=3, support_per_class=5,
                 total_query_per_episode=15, episodes_per_step=8):
        if way != 3 or total_query_per_episode <= 0 or total_query_per_episode % way:
            raise ValueError("Training requires 3-way with a positive balanced Query count")
        self.pools = class_indices
        self.fake_classes = sorted(set(class_indices) - {"real"})
        if len(self.fake_classes) != 32:
            raise ValueError("Training requires exactly 32 Fake classes")
        self.support = support_per_class
        self.query = total_query_per_episode // way
        self.episodes_per_step = episodes_per_step
        for name, pool in self.pools.items():
            if len(pool) < self.support + self.query or len(pool) != len(set(pool)):
                raise ValueError(f"Insufficient or duplicate samples for {name}")

    def sample_episode(self):
        classes = ("real", *random.sample(self.fake_classes, 2))
        support, query, sl, ql = [], [], [], []
        for label, name in enumerate(classes):
            selected = random.sample(self.pools[name], self.support + self.query)
            support.extend(selected[:self.support])
            query.extend(selected[self.support:])
            sl.extend([label] * self.support)
            ql.extend([label] * self.query)
        return Episode(classes, tuple(support), tuple(query), tuple(sl), tuple(ql))

    def sample_step(self):
        return [self.sample_episode() for _ in range(self.episodes_per_step)]
