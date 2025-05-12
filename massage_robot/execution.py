class GridExecutor:
    """
    Executes massage strokes over a 2D grid of regions.
    """
    def __init__(self, path_planner, controller, safety):
        self.planner = path_planner
        self.controller = controller
        self.safety = safety

    def execute_grid(self, region_map, env):
        results = {}
        for i, row in enumerate(region_map):
            for j, region_id in enumerate(row):
                wpts = self.planner.plan_waypoints(region_id)
                for wp in wpts:
                    tau = self.controller.compute(env.read_force())
                    obs = env.step(wp, tau)
                    ok, msg = self.safety.check(obs['contact_forces'].values(),
                                               [v for _, v in obs['joint_states']])
                    if not ok:
                        raise RuntimeError(f"Safety stop at region {region_id}: {msg}")
                results[(i,j)] = 'completed'
        return results
