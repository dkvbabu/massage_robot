import pybullet as p
import pybullet_data
from torch.utils.tensorboard import SummaryWriter

class MassageEnv:
    """
    Core simulation environment for autonomous robotic massage.
    - Initializes physics client
    - Loads URDFs for robot arm, end-effector, bench, and human phantom
    - Steps simulation and provides state+sensor feedback
    - Provides methods for IK and sensor emulation
    """
    def __init__(self, gui=True, time_step=1./240., log_dir="runs/env_logs"):
        self.gui = gui
        if gui:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setTimeStep(time_step)
        self._load_models()
        self.step_count = 0
        self.writer = SummaryWriter(log_dir)

    def _load_models(self):
        # TODO: replace with actual URDF file paths
        self.plane = p.loadURDF("plane.urdf")
        self.robot  = p.loadURDF("7dof_arm.urdf", useFixedBase=True)
        self.bench  = p.loadURDF("bench.urdf", [0,0,0], useFixedBase=True)
        self.phantom= p.loadURDF("human_phantom.urdf", [0.5,0,0], useFixedBase=True)
        # assume end-effector is last link
        self.end_effector_link = p.getNumJoints(self.robot) - 1

    def calculate_ik(self, position, orientation):
        """
        Compute joint angles to reach a desired end-effector pose.
        """
        return p.calculateInverseKinematics(
            self.robot,
            self.end_effector_link,
            targetPosition=position,
            targetOrientation=orientation
        )

    def _read_force_sensors(self):
        """
        Emulate force sensors by summing contact normal forces per robot link.
        """
        contacts = p.getContactPoints(bodyA=self.robot, bodyB=self.phantom)
        forces = {}
        for c in contacts:
            link_idx = c[3]  # link index on robot
            force = c[9]     # normal force magnitude
            forces[link_idx] = forces.get(link_idx, 0) + force
        return forces

    def step(self, joint_commands, render=True):
        """
        Apply joint_commands (list of torques or angles), step simulation,
        and return observation dict.
        """
        for i, cmd in enumerate(joint_commands):
            p.setJointMotorControl2(
                bodyIndex=self.robot,
                jointIndex=i,
                controlMode=p.POSITION_CONTROL,
                targetPosition=cmd
            )
        p.stepSimulation()
        obs = {
            "joint_states": [ (js[0], js[1]) 
                              for js in p.getJointStates(self.robot, range(p.getNumJoints(self.robot))) ],
            "contact_forces": self._read_force_sensors(),
        }
        for idx, (pos, vel) in enumerate(obs["joint_states"]):
            self.writer.add_scalar(f"joint/{idx}/position", pos, self.step_count)
            self.writer.add_scalar(f"joint/{idx}/velocity", vel, self.step_count)
        for link, force in obs["contact_forces"].items():
            self.writer.add_scalar(f"force/link_{link}", force, self.step_count)
        self.step_count += 1
        if render and self.gui:
            pass
        return obs

    def reset(self):
        p.resetSimulation()
        self._load_models()
        self.step_count = 0
        return self.step([], render=False)

    def disconnect(self):
        p.disconnect()
