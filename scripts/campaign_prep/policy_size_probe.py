"""P2 Step 5 policy-size probe: ONE 2000-timestep PPO training per task, saving the SB3 model
exactly as the campaign would save a policy, to measure its on-disk size.

Mirrors evodevo.evaluation.ppo_runner.evaluate_body's own construction verbatim
(make_vec_env(task, n_envs=1, seed, env_kwargs={body, connections}); PPO("MlpPolicy", vec_env,
seed=seed, verbose=0); model.learn(total_timesteps)) -- ppo_runner itself is NOT modified and
does not expose the trained model, hence this standalone measurement script.

Body: the LARGEST possible 5x5 morphology (all 25 voxels filled, alternating H/V actuators),
because an SB3 MlpPolicy's parameter count grows with the body's observation/action dimensions
(point masses, actuator count) and is independent of the number of timesteps trained -- so the
full body is the conservative upper bound for every campaign body on that task.
"""
import json, os, sys, time
sys.path.insert(0, os.getcwd())

OUT = "results/campaign_prep/disk"
TIMESTEPS = 2000


def main():
    import numpy as np
    import evogym.envs  # noqa: F401
    from evogym import get_full_connectivity
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    from evodevo.morphology.validate import validate_body

    body = np.array([[3 if (r + c) % 2 == 0 else 4 for c in range(5)] for r in range(5)])
    assert validate_body(body).valid
    connections = get_full_connectivity(body)
    out = {"timesteps_requested": TIMESTEPS, "body": body.tolist(), "tasks": {}}
    for task in ("Walker-v0", "UpStepper-v0"):
        t0 = time.time()
        vec_env = make_vec_env(task, n_envs=1, seed=0, env_kwargs={"body": body, "connections": connections})
        model = PPO("MlpPolicy", vec_env, seed=0, verbose=0)
        model.learn(total_timesteps=TIMESTEPS)
        path = os.path.join(OUT, f"policy_probe_{task}.zip")
        model.save(path)
        n_params = int(sum(p.numel() for p in model.policy.parameters()))
        out["tasks"][task] = {"zip_bytes": os.path.getsize(path), "policy_parameters": n_params,
                              "obs_dim": int(vec_env.observation_space.shape[0]),
                              "action_dim": int(vec_env.action_space.shape[0]),
                              "timesteps_trained": int(model.num_timesteps), "wall_s": time.time() - t0}
        vec_env.close()
        print(task, out["tasks"][task], flush=True)
    json.dump(out, open(os.path.join(OUT, "policy_size_probe.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
