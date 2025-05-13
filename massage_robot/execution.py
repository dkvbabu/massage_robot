## massage_robot/execution.py
class GridExecutor:
    """
    Executes massage strokes over a 2D region grid.
    """
    def __init__(self, planner, controller, safety, env):
        self.planner = planner
        self.controller = controller
        self.safety = safety
        self.env = env

    def execute(self, grid_map, region_vertices_map, samples_per_region=100, dt=1/240):
        # Pre-plan full trajectory
        path = self.planner.plan_grid(grid_map, region_vertices_map, samples_per_region)
        results = []
        for pos in path:
            # IK to joint angles
            joint_targets = self.env.calculate_ik(pos.tolist(), self.env.default_orientation)
            # read current force & velocity
            forces = self.env._read_force_sensors().values()
            velocities = [v for _,v in self.env.step(joint_targets, render=False)['joint_states']]
            # safety
            ok, msg = self.safety.verify(forces, velocities)
            if not ok:
                raise RuntimeError(f"Safety stop: {msg}")
            # compute torque
            curr_force = max(forces)
            tau = self.controller.compute_torque(curr_force, dt)
            # apply torque offset
            cmds = [jt + tau for jt in joint_targets]
            obs = self.env.step(cmds)
            results.append(obs)
        return results
