import os
import sys
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from gym import Wrapper
import utils.record

class EasyLunar(Wrapper):
    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        #if self.env._elapsed_steps == self.env._max_episode_steps:
        #    print("max")
        #    reward = -100
        if abs(observation[2]) < 0.0001 and abs(observation[3]) < 0.0001 \
                and observation[6] and observation[7] \
                and abs(observation[0] - observation[8]) < 0.2:
            done = True
            reward = +100
        return observation, reward, done, info

class TimeLimit(Wrapper):    
    def __init__(self, env, max_episode_steps=None):    
        super(TimeLimit, self).__init__(env)    
        if max_episode_steps is None and self.env.spec is not None:    
            max_episode_steps = env.spec.max_episode_steps    
        if self.env.spec is not None:    
            self.env.spec.max_episode_steps = max_episode_steps    
        self._max_episode_steps = max_episode_steps    
        self._elapsed_steps = None    
     
    def step(self, action):    
        assert self._elapsed_steps is not None, "Cannot call env.step() before calling reset()"    
        self._elapsed_steps += 1    
        if self._elapsed_steps >= self._max_episode_steps:    
            print(1, self.env.game_over)
            self.env.game_over = True
            print(2, self.env.game_over)

        observation, reward, done, info = self.env.step(action)    

        if self._elapsed_steps >= self._max_episode_steps:    
            info['TimeLimit.truncated'] = not done    
            done = True    
            print(reward)
            print(3, self.env.game_over)
        return observation, reward, done, info    
     
    def reset(self, **kwargs):    
        self._elapsed_steps = 0    
        return self.env.reset(**kwargs)    


class TestWrapper(Wrapper):
    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        if done:
            print('done',reward)

        return observation, reward, done, info

class Record(Wrapper):
    memory_labels = ('observation', 'reward', 'done', 'info', 'action', 'img')
    def __init__(self, env, save_img=False):
        super(Record, self).__init__(env)
        self.save_img = save_img
        self.memory = None

    def reset(self, **kwargs):
        if self.memory is None:
            self.memory = {label: list() for label in self.memory_labels} 
        observation = self.env.reset(**kwargs)
        self.memory['observation'].append(observation.tolist())
        return observation

    def snapshot(self):
        import io
        import PIL
        img = self.env.render(mode='rgb_array')
        img = PIL.Image.fromarray(img)
        with io.BytesIO() as output:
            img.save(output, 'PNG')
            img = output.getvalue()
        return img

    """
    def render(self, **kwargs):
        import io
        import PIL
        if self.memory is None:
            self.memory = {label: list() for label in self.memory_labels} 
        img = self.env.render(mode='rgb_array', **kwargs)
        img = PIL.Image.fromarray(img)
        with io.BytesIO() as output:
            img.save(output, 'PNG')
            img = output.getvalue()
        self.memory['img'].append(img)
        return img
    """

    def step(self, action):
        # rtn: observation, reward, done, info 
        rtn = self.env.step(action)

        self.memory['action'].append(action)
        for label, value in zip(self.memory_labels[:-2], rtn):
            if type(value) is np.ndarray:
                self.memory[label].append(value.tolist())
            else:
                self.memory[label].append(value)
        if self.save_img:
            self.memory['img'].append(self.snapshot())

        # if done
        if rtn[2]:
            # done
            utils.record.add(self.memory)
            self.memory = None
        return rtn
