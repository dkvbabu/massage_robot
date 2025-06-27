

# Autonomous Robotic System for Massage

*Authors: Yasin Yousif (https://www.rlbyexample.net/), Venkatesh Babu (https://github.com/dkvbabu)*

*Open Source: https://github.com/fellowship/massage_robot*

## Introduction

Back and neck pains are widespread conditions, affecting approximately 60% to 80% adults in their lifetime [[source]](https://journals.lww.com/md-journal/fulltext/2017/05190/trends_in_diagnosis_of_painful_neck_and_back.3.aspx). These discomforts can arise from various factors, ranging from simple 
muscle tension related to prolonged work or poor posture, to more serious underlying conditions like spinal displacement or age-related bone weakening, particularly in older adults.

Many individuals experiencing these issues seek relief through professional massage therapy. While skilled massage therapists can provide significant benefits, the expertise required is often scarce and valuable. **This creates an opportunity for robotic automation to assist**, specifically through the use of specialized robotic arms to perform massage. This idea  has motivated innovation, with startups like [aescape](https://www.aescape.com/) developing commercial robotic massage benches designed to offer consistent, accessible treatment with efficiency comparable to human massage therapists, as shown in the image below.

<center>
<image width="75%" src="./media/aescape.webp"/>
</br>
<caption>
Aescape robotic massage bench. Source: www.aescape.com
</caption>
</center>



Despite the remarkable advancements in robotics and artificial intelligence, developing a truly effective robotic massage system remains a significant challenge. Because replicating the exact movements and appropriate pressure required to massage human muscle, ensuring safety, and avoiding potential failure points are complex tasks. The work here focuses on designing and testing a foundational robotic massaging system within a simulation environment.

To illustrate our approach, consider a robotic arm equipped with a 3-degree-of-freedom (3-DoF) arm and a flexible end effector, positioned above a human model. The system's initial steps involve locating the human model, reaching a designated starting point, and then executing a calibrated massage motion.  A continuous feedback loop is essential to proactively prevent potential issues and adapt to the model's response.

Even this simplified demonstration highlights several intricate steps. Therefore, we propose a hybrid solution that combines a rule-based path-following method with a Reinforcement Learning (RL) approach trained within a simulated environment.  Our framework leverages tools like [Pybullet](https://pybullet.org/wordpress/), and  carefully defines the state, action, and reward formulations – critical components for successful RL training.

To rigorously evaluate our design, we benchmarked multiple RL methods across various environment configurations. This included comparisons of systems with and without reinforcement learning, as well as those lacking path generation capabilities. The results of these comparisons are then discussed and detailed in our findings.


Our tests conclude that this combined approach, specifically when utilizing the Deep Deterministic Policy Gradient (DDPG) RL method, achieves the best overall performance, demonstrating superior pressure control and high episodic reward.  In summary, this project has yielded the following key achievements:

*   Proposing an effective methodology for training robotic massaging arms.
*   Benchmarking a variety of configurations, exploring different state/action/reward formulations and RL methodologies.
*   Extensively testing trained models across diverse massage paths and under varying conditions, including the introduction of random noise.


## Methodology

In the following, we break down the core components of a PyBullet-based robotic massage simulation that integrates a detailed human phantom model, a UR5 robotic arm, a path generation for natural motion, and an RL  framework with well-defined states, actions, and rewards.

### Simulation Environment

The simulation environment is built on PyBullet, a powerful physics engine that enables real-time rigid body dynamics and collision detection. The environment initializes with a plane, a cube, and a UR5 robotic arm loaded from its URDF description. Gravity and timestep settings mimic real-world physics, ensuring realistic robot and human interactions.
A human phantom model is loaded and positioned within the scene, serving as the “patient.” The environment manages the robot’s controlled joints and end effector, stepping through simulation frames and collecting detailed contact data. This setup provides a rich platform for training and evaluating robotic massage strategies.

At the heart of the simulation is a sophisticated human phantom model imported from the project: [gym-assistant](https://github.com/Healthcare-Robotics/assistive-gym) constructed from capsules and spheres representing limbs and joints. The model supports male and female variants with configurable mass, size, and skin tone, enhancing realism.

The robotic therapist in this simulation is a UR5 arm, a widely used industrial manipulator known for its precision and flexibility. Loaded from a URDF file, based on [robot-description](https://github.com/robot-descriptions/robot_descriptions.py). The arm features six controlled joints and an end effector designed to perform massage strokes.
Inverse kinematics computes joint angles to reach target positions along generated paths, while position control with force limits ensures safe, smooth movements. This precise control framework allows the robot to execute complex, adaptive massage patterns in close proximity to the human model.

<center>
<image width="60%" src="./media/massage_y.gif"/>
<caption>
</br>
Full simulated scene
</caption>
</center>

### Mixed Control (Reinforcement Learning + Path Generation)

To mimic the fluidity of human massage, the system generates sinusoidal trajectories between points near the human model’s surface. These trajectories are cyclically followed by the robot’s end effector, with calibrated updates (by the RL model) adapting to the human’s position and surface contours.
This approach produces continuous, naturalistic motion patterns that leads to maintaining consistent contact and pressure, essential for effective massage. The full workflow is as depicted in the figure below.

<center>
<image src="./media/plan1.png"/>
<caption>
</br>
Full Workflow of the Massage System
</caption>
</center>




For the reinforcement learning step, a comprehensive state space, action space, and reward function should be defined. In our case, we found the following setup efficient: 

<center>
<image width="65%" src="./media/envstates.png"/>
</br>
<caption>
State/Action/Reward Setup for our RL training step.
</caption>
</center>

- State: Encapsulates contact point coordinates, force magnitudes, normal distances and directions, end effector pose and orientation, as well as joint positions and velocities. 

- Action: Specifies the **change in the target** position for the robot’s end effector from the generated path, to adjust the pressure of the trajectory.

- Reward: Encourages the robot to apply appropriate contact pressure by rewarding forces below a maximum threshold (50 Newton) and penalizing excessive force or incorrect contact points. This reward structure guides the agent toward safe, effective massage behaviors.

This PyBullet-based robotic massage simulation elegantly combines physics-based modeling, detailed human anatomy, precise robotic control, and reinforcement learning principles.

For RL task, we explore three state-of-the-art RL algorithms applied to the simulation: **DDPG (Deep Deterministic Policy Gradient), PPO (Proximal Policy Optimization), and TD3 (Twin Delayed Deep Deterministic Policy Gradient)**. We also highlight key aspects of their training processes and implementation details. The implantation of these algorithm are based on [CleanRL code base](https://github.com/vwxyzjn/cleanrl). 

Each algorithm offers unique strengths, and their implementations highlight best practices in training, exploration, and stability. In practice, it is wise to benchmark all of them and choose the most suitable one for our use case, which is what we have done here.

## Evaluation and Results
In the following we show two types of curves for the full experiments done in this project. First the training curves resulted from tensorboard logging. Second, the pressure curve illustrating the arm normal force on the model.

Finally, a complete table for the performance of all methods is shown to determine the model of best performance.

### Training Curves 
	 	 	 	
The training curve for the DDPG-Y model illustrates the learning progress of the robotic massage system for the case of following a massage path along the Y axis of the simulation model. The Y-axis of the graph represents the episodic return, which measures the cumulative reward obtained by the agent during each training episode. The X-axis represents the training iterations. This curve below provides insight into how effectively the model improves its performance over time in this specific task.

<center>
<image width="100%" src="./media/DDPG_Y_tf.png"/>
</br>
<caption>
DDPG Training curve massaging along the height of the model
</caption>
</center>


This next training curve below compares the performance of the DDPG and PPO algorithms when trained on a massage path along the X axis of the simulation environment, with no noise introduced during training. The graph highlights the learning efficiency and stability of both methods under ideal, noise-free conditions.

<center>
<image width="100%" src="./media/DDPG+PPO_tf.png"/>
</br>
<caption>
DDPG and PPO Training curve massaging along the width of the model.
</caption>
</center>

The training curves below for DDPG and PPO algorithms under noisy conditions shows the impact of environmental noise on learning speed along the X axis massage path (body width). The plot X-axis again represents episodic return, but the presence of noise during training results in generally lower returns, indicating the increased difficulty the models face in adapting to uncertain and variable conditions.

<center>
<image width="100%" src="./media/DDPG+PPO_tf_noise.png"/>
</br>
<caption>
DDPG and PPO Training curve massaging along the width of the model with random initialization.
</caption>
</center>

> Overall, the training curves demonstrate that **DDPG achieves the fastest and most stable learning progress**, especially in noise-free environments, while the introduction of random initialization significantly challenges both DDPG and PPO, reducing their episodic returns and slowing their convergence.

### Pressure Curves

The shown pressure curve over 360 simulation steps below shows the contact force applied by the robotic end effector on the human model. Throughout most of the episode, the force remains below 50 Newtons, which is within a safe and comfortable range for massage. A notable spike occurs near the end (when exceeding the 360 step horizon), indicating a brief increase in applied pressure that shows a room for improvement to avoid discomfort or injury.
	 	 	 	
Additionally, the graphs here shows right under the pressure curve, a contact plot (the end effector index 7 indicates direct contact), the human model part contact index, and the actual Z-value of the planned (blue) and  modified (orange) massage path.

<center>
<image width="100%" src="./media/DDPG_pressure.png"/>
</br>
<caption>
DDPG pressure and contact curves over 720 steps (without noise along model width)
</caption>
</center>

The pressure curve for the DDPG-Y model over 360 steps reveals that the robotic arm maintains contact forces predominantly under 50 Newtons, with occasional interruptions. These fluctuations suggest moments where the robot adjusts its pressure or loses contact briefly, reflecting the dynamic nature of the massage path along the Y axis.

<center>
<image width="100%" src="./media/DDPG_Y_pressure.png"/>
</br>
<caption>
DDPG pressure and contact curves over 360 steps (without noise along body height)
</caption>
</center>

The table below presents the mean and standard deviation of episodic returns over 100 samples for various reinforcement learning methods under both stable and noisy environmental conditions. The methods benchmarked include no reinforcement learning (No RL), PPO, DDPG, DDPG trained along the Y axis (DDPG-Y), and TD3 without path guidance. 

**Results indicate that DDPG and DDPG-Y achieve the highest returns in stable environments, while noisy conditions generally reduce performance across all methods. Notably, DDPG maintains relatively strong performance even with noise, demonstrating robustness. TD3 fails without using the sine waves as basis, reaching only 3300 as mean return. This shows the efficiency of our hybrid proposed model above**

<center>
<caption>
Mean and standard deviation of 100 samples (returns) for all massage control variants
</caption>
</br>
<image width="100%" src="./media/final_benchmark.png"/>
</center>

Finally, in the following, we show two demonstrating videos of the simulation scenarios utilizing DDPG along the height of the model (first video), then along the width of the model (second video), corresponding to the trained models above without noise.

<center>
<image width="70%" src="./media/massage_y.gif"/>
</br>
<caption>
DDPG model performing massage along the height of the human model (Y world axis)
</caption>
</center>

<center>
<image width="70%" src="./media/massage_x.gif"/>
</br>
<caption>
DDPG model performing massage along the width of the human model (X world axis)
</caption>
</center>

## Conclusion

Our experimental results validate the efficacy of our approach, particularly when coupled with value-based methods like **DDPG**. However, we observed a temporary slowdown in learning and a slight performance dip when training in noisy environments (with random initial position for the human model and random frequency and amplitude for the massage path). To address this, we believe **curriculum learning** – gradually introducing noise during training, starting with stable conditions and progressively increasing complexity – holds significant potential to accelerate the learning process.

While promising, our current models exhibit *limitations*. Their training, conducted without exposure to varied environmental conditions, renders them somewhat sensitive to changes in model or arm positioning. Furthermore, a crucial safety feature – the ability to execute emergency maneuvers to avoid potential harm to the human – remains unimplemented in this work. A straightforward solution to this would involve triggering a pre-programmed avoidance maneuver whenever the force exerted exceeds a defined threshold (e.g., 50 Newtons).

Looking ahead, several avenues for future improvement are readily apparent. Exploring alternative end effectors beyond the flexible ball would broaden the range of possible massage techniques. A humanoid hand with multiple degrees of freedom, for example, would offer a more adaptable approach.  Moreover, the adoption of a dual-arm configuration, mirroring systems like the Aesape bench, represents a compelling direction for future research. Finally, personalizing the maximum applied force based on individual preferences is essential.  This will necessitate either developing a more customized model or training multiple reinforcement learning models for different pressure levels.
