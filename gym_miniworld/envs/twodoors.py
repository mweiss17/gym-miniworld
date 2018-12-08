import numpy as np
import math
from ..miniworld import MiniWorldEnv, Room
from ..entity import Door, ImageFrame

class TwoDoorsEnv(MiniWorldEnv):
    """
    Environment in which the goal is to go to a red box
    placed randomly in one big room.
    """

    def __init__(self, size=10, **kwargs):
        assert size >= 2
        self.size = size

        super().__init__(
            max_episode_steps=180,
            **kwargs
        )

    def _gen_world(self):
        room = self.add_rect_room(
            min_x=0,
            max_x=self.size,
            min_z=0,
            max_z=self.size,
            floor_tex='concrete',
            wall_tex='picket_fence',
            wall_height=1.1,
            no_ceiling=True
        )

        import math
        self.door1 = ImageFrame(pos=[2., 0.9, 0.], dir=-1.66, tex_name="metal_door", width=1.)
        self.entities.append(self.door1)
        self.door2 = ImageFrame(pos=[8., 0.9, 0.], dir=-1.66, tex_name="metal_door", width=1.)
        self.entities.append(self.door2)
        self.svhn1 = ImageFrame(pos=[3.0, 1.5, 0.], dir=-1.66, tex_name="svhn", width=1.)
        self.entities.append(self.svhn1)

        self.svhn2 = ImageFrame(pos=[9.0, 1.5, 0.], dir=-1.66, tex_name="svhn2", width=1.)
        self.entities.append(self.svhn2)

        # self.door2 = self.place_entity(Door(color='blue'), pos=[6.66, 0., 0.])
        #self.entities.append(ImageFrame(pos=[3.33, 0., 0.], dir=0, tex_name="svhn_1", width=2.))
        self.entities.append(ImageFrame(pos=[0., 1.7, 5.], dir=0., tex_name="street_scene", width=10.))
        # self.entities.append(ImageFrame(pos=[3., 1.7, 5.], dir=0., tex_name="svhn_1", width=2.))
        self.place_agent()

    def step(self, action):
        obs, reward, done, info = super().step(action)

        if self.near(self.door1):
            reward += self._reward()
            done = True

        return obs, reward, done, info
