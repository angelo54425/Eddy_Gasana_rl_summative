import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

import environment
from environment.career_env import CareerPathEnv

env = CareerPathEnv(steps=60, debug=True)   # debug=True will print internals
obs, _ = env.reset()
print("RESET OBS:", obs)

for i in range(60):
    a = env.action_space.sample()
    obs, r, t, tr, info = env.step(a)
    print(f"[{i}] action={a}, reward={r:.3f}, obs={obs}, info={info}")
    if t or tr:
        print("EPISODE ENDED:", info)
        break

env.close()
