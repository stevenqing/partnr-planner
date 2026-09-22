"""Opt-in switch `+iclr_env_over_metrics=True` for the B repo's planner_demo_seeded.py.

Problem (G0 smoke 09-22): when the habitat env ends an episode on its own (task success or the step
limit), the evaluation loop keeps stepping, env.step raises "Episode over, call reset before calling
step", and the episode is written as a failure with no metrics and no planner-log, i.e. it drops out
of the per-episode averages even when the task was completed.

With the switch on: once the habitat Env reports episode_over, EnvironmentInterface.step is no longer
forwarded to the env (it returns the last real step's (obs, reward, done, info)), and the runner's
next get_low_level_actions returns should_end=True, so run_instruction takes its normal end-of-episode
branch: measures are updated on the state at which the env ended and metrics / planner-log are written.
Nothing else changes. With the switch off this module is never imported.
"""


def install(eval_runner, env_interface):
    state = {"over": False, "last": None}
    orig_step = env_interface.step
    orig_get = eval_runner.get_low_level_actions
    orig_run = eval_runner.run_instruction

    def step(low_level_actions):
        hab_env = env_interface.env.env.env._env
        if hab_env.episode_over and state["last"] is not None:
            if not state["over"]:
                print("ICLR_ENV_OVER: habitat env ended the episode; ending it and recording metrics", flush=True)
            state["over"] = True
            return state["last"]
        out = orig_step(low_level_actions)
        state["last"] = out
        return out

    def get_low_level_actions(*args, **kwargs):
        low_level_actions, planner_info, should_end = orig_get(*args, **kwargs)
        return low_level_actions, planner_info, (should_end or state["over"])

    def run_instruction(*args, **kwargs):
        state["over"], state["last"] = False, None
        return orig_run(*args, **kwargs)

    env_interface.step = step
    eval_runner.get_low_level_actions = get_low_level_actions
    eval_runner.run_instruction = run_instruction
    print("ICLR_ENV_OVER switch installed", flush=True)
