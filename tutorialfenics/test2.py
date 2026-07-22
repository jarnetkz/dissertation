import numpy as np

dt = 0.01 
t = 0
step = 0
T = 5*2*np.pi
target_times = [
    2*np.pi + np.pi/2,
    4*np.pi + np.pi/2
]

target_steps = [round(ti / dt) for ti in target_times]
print("target_steps:", target_steps)
while t < T + 1e-12:
    
    if step in target_steps:
        # saved_norms[round(t, 10)] = {
        #     "pressure_l2": norm(p_sol, "L2"),
        #     "concentration_l2": norm(c_sol, "L2")
        # }

        print("Saved at step =", step, "time =", t)
        
        
    t += dt
    step += 1