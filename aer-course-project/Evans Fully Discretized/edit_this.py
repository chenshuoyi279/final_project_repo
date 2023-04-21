"""Write your proposed algorithm.
[NOTE]: The idea for the final project is to plan the trajectory based on a sequence of gates 
while considering the uncertainty of the obstacles. The students should show that the proposed 
algorithm is able to safely navigate a quadrotor to complete the task in both simulation and
real-world experiments.

Then run:

    $ python3 final_project.py --overrides ./getting_started.yaml

Tips:
    Search for strings `INSTRUCTIONS` and `REPLACE THIS (START)` in this file.

    Change the code between the 5 blocks starting with
        #########################
        # REPLACE THIS (START) ##
        #########################
    and ending with
        #########################
        # REPLACE THIS (END) ####
        #########################
    with your own code.

    They are in methods:
        1) planning
        2) cmdFirmware

"""
# source ~/anaconda3/etc/profile.d/conda.sh
# then conda activate aer1217-project
import time 
import numpy as np
import math
from collections import deque
from Search_based_Planning.Search_2D import  env,Astar
try:
    from project_utils import Command, PIDController, timing_step, timing_ep, plot_trajectory, draw_trajectory
except ImportError:
    # PyTest import.
    from .project_utils import Command, PIDController, timing_step, timing_ep, plot_trajectory, draw_trajectory

#########################
# REPLACE THIS (START) ##
#########################

# Optionally, create and import modules you wrote.
# Please refrain from importing large or unstable 3rd party packages.
# try:
#     import example_custom_utils as ecu
# except ImportError:
#     # PyTest import.
#     from . import example_custom_utils as ecu

#########################
# REPLACE THIS (END) ####
#########################
def distance(point1, point2):
        print(point1, point2)
        return ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) ** 0.5
class Controller():
    """Template controller class.

    """

    def __init__(self,
                 initial_obs,
                 initial_info,
                 use_firmware: bool = False,
                 buffer_size: int = 100,
                 verbose: bool = False
                 ):
        """Initialization of the controller.

        INSTRUCTIONS:
            The controller's constructor has access the initial state `initial_obs` and the a priori infromation
            contained in dictionary `initial_info`. Use this method to initialize constants, counters, pre-plan
            trajectories, etc.

        Args:
            initial_obs (ndarray): The initial observation of the quadrotor's state
                [x, x_dot, y, y_dot, z, z_dot, phi, theta, psi, p, q, r].
            initial_info (dict): The a priori information as a dictionary with keys
                'symbolic_model', 'nominal_physical_parameters', 'nominal_gates_pos_and_type', etc.
            use_firmware (bool, optional): Choice between the on-board controll in `pycffirmware`
                or simplified software-only alternative.
            buffer_size (int, optional): Size of the data buffers used in method `learn()`.
            verbose (bool, optional): Turn on and off additional printouts and plots.

        """
        # Save environment and control parameters.
        self.CTRL_TIMESTEP = initial_info["ctrl_timestep"]
        self.CTRL_FREQ = initial_info["ctrl_freq"]
        self.initial_obs = initial_obs
        self.VERBOSE = verbose
        self.BUFFER_SIZE = buffer_size

        # Store a priori scenario information.
        # plan the trajectory based on the information of the (1) gates and (2) obstacles. 
        self.NOMINAL_GATES = initial_info["nominal_gates_pos_and_type"]
        self.NOMINAL_OBSTACLES = initial_info["nominal_obstacles_pos"]

        # Check for pycffirmware.
        if use_firmware:
            self.ctrl = None
        else:
            # Initialize a simple PID Controller for debugging and test.
            # Do NOT use for the IROS 2022 competition. 
            self.ctrl = PIDController()
            # Save additonal environment parameters.
            self.KF = initial_info["quadrotor_kf"]

        # Reset counters and buffers.
        self.reset()
        self.interEpisodeReset()

        # perform trajectory planning
        t_scaled = self.planning(use_firmware, initial_info)

        ## visualization
        # Plot trajectory in each dimension and 3D.
        plot_trajectory(t_scaled, self.waypoints, self.ref_x, self.ref_y, self.ref_z)

        # Draw the trajectory on PyBullet's GUI.
        draw_trajectory(initial_info, self.waypoints, self.ref_x, self.ref_y, self.ref_z)

    def discrete_coordinates(self,x, y,grid_size=0.05):
        x_discrete = int((x + 3.5) // grid_size)
        y_discrete = int((y + 3.5) // grid_size)
        return x_discrete, y_discrete
    
    def gate_obstacle_coordinates(self, x, y, yaw):
        obstacle1_x = x + 0.2 * math.cos(yaw + math.pi / 2)
        obstacle1_y = y + 0.2 * math.sin(yaw + math.pi / 2)
        obstacle2_x = x + 0.2 * math.cos(yaw - math.pi / 2)
        obstacle2_y = y + 0.2 * math.sin(yaw - math.pi / 2)
        return (obstacle1_x, obstacle1_y), (obstacle2_x, obstacle2_y)
    
    def discretize_obstacle(self,x, y, diameter,grid_size=0.05):
        x_discrete, y_discrete = self.discrete_coordinates(x, y)
        obstacle_points = []
        for i in range(-int(diameter // grid_size), int(diameter // grid_size) + 1):
            for j in range(-int(diameter // grid_size), int(diameter // grid_size) + 1):
                obstacle_points.append((x_discrete + i, y_discrete + j))
        return obstacle_points
   
    def planning(self, use_firmware, initial_info):
        """Trajectory planning algorithm"""
        #########################
        # REPLACE THIS (START) ##
        #########################
        ## generate waypoints for planning

        # initial waypoint where it takes off
        if use_firmware:
            waypoints = [(self.initial_obs[0], self.initial_obs[2], initial_info["gate_dimensions"]["tall"]["height"])]  # Height is hardcoded scenario knowledge.
        else:
            waypoints = [(self.initial_obs[0], self.initial_obs[2], self.initial_obs[4])]
        t1 = time.perf_counter()
#get gate obstacles


        gate_obstacle_diameter = 0.1
        discretized_gate_obstacles = []
        for gate in initial_info['nominal_gates_pos_and_type']:
            x, y, _, _, _, yaw, _ = gate
            obstacle1, obstacle2 = self.gate_obstacle_coordinates(x, y, yaw)
            discretized_obstacle1 = self.discretize_obstacle(*obstacle1, gate_obstacle_diameter)
            discretized_obstacle2 = self.discretize_obstacle(*obstacle2, gate_obstacle_diameter)
            discretized_gate_obstacles.extend(discretized_obstacle1 + discretized_obstacle2)
        grid_size=0.05
#get obstacles
        obstacles=initial_info['nominal_obstacles_pos']
        obstacle_diameter = 0.12
        discretized_obstacles = []
        for obstacle in obstacles:
            x, y, _, _, _, _ = obstacle
            x_discrete, y_discrete = self.discrete_coordinates(x, y)
            # Add discretized obstacle coordinates considering the diameter
            for i in range(-int(obstacle_diameter // grid_size), int(obstacle_diameter // grid_size) + 1):
                for j in range(-int(obstacle_diameter // grid_size), int(obstacle_diameter // grid_size) + 1):
                    discretized_obstacles.append((x_discrete + i, y_discrete + j))
    

        all_discretized_obstacles = discretized_obstacles + discretized_gate_obstacles
        print("discretized_obstacles", discretized_obstacles)
        s_start = (self.initial_obs[0], self.initial_obs[2])
        paths = []
        gate_waypoints = []
        gate_count = len(initial_info['nominal_gates_pos_and_type'])
        # Rearrange the sequence of the gates
        gates_sequence = list(range(gate_count))
        gates_sequence = [0,1,3,2]  # Update this if the number of gates changes
        # Iterate through the rearranged sequence of gates
        for index in gates_sequence:
            gate = initial_info['nominal_gates_pos_and_type'][index]
            x, y, _, _, _, yaw, _ = gate
            # Update distance_between_gate for the second last gate
            if index == gate_count - 4:
                distance_between_gate = 0.3
            else:
                distance_between_gate = 0.3

            waypoint1_x = x + distance_between_gate * math.cos(yaw + math.pi / 2)
            waypoint1_y = y + distance_between_gate * math.sin(yaw + math.pi / 2)
            waypoint2_x = x + distance_between_gate * math.cos(yaw - math.pi / 2)
            waypoint2_y = y + distance_between_gate * math.sin(yaw - math.pi / 2)
            gate_waypoints.append(((waypoint1_x, waypoint1_y), (waypoint2_x, waypoint2_y)))

        current_point = s_start
        rearranged_waypoints = []

        for wp_pair in gate_waypoints:
            wp1, wp2 = wp_pair

            if distance(current_point, wp1) < distance(current_point, wp2):
                rearranged_waypoints.extend([wp1, wp2])
                current_point = wp2
            else:
                rearranged_waypoints.extend([wp2, wp1])
                current_point = wp1
        
        rearranged_waypoints.extend([(-0.5,2.0)])

        print("rearranged_waypoints", rearranged_waypoints)
        #all_discretized_obstacles is generated earlier from this code, pasted here to save time
        all_discretized_obstacles=   [(97, 17), (97, 18), (97, 19), (97, 20), (97, 21), (98, 17), 
                                      (98, 18), (98, 19), (98, 20), (98, 21), (99, 17), (99, 18), 
                                      (99, 19), (99, 20), (99, 21), (100, 17), (100, 18), (100, 19), 
                                      (100, 20), (100, 21), (101, 17), (101, 18), (101, 19), (101, 20),
                                        (101, 21), (77, 47), (77, 48), (77, 49), (77, 50), (77, 51), 
                                        (78, 47), (78, 48), (78, 49), (78, 50), (78, 51), (79, 47), (79, 48), 
                                        (79, 49), (79, 50), (79, 51), (80, 47), (80, 48), (80, 49), (80, 50), 
                                        (80, 51), (81, 47), (81, 48), (81, 49), (81, 50), (81, 51), (97, 67), 
                                        (97, 68), (97, 69), (97, 70), (97, 71), (98, 67), (98, 68), (98, 69), 
                                        (98, 70), (98, 71), (99, 67), (99, 68), (99, 69), (99, 70), (99, 71), 
                                        (100, 67), (100, 68), (100, 69), (100, 70), (100, 71), (101, 67), (101, 68), 
                                        (101, 69), (101, 70), (101, 71), (47, 67), (47, 68), (47, 69), (47, 70), 
                                        (47, 71), (48, 67), (48, 68), (48, 69), (48, 70), (48, 71), (49, 67),
                                          (49, 68), (49, 69), (49, 70), (49, 71), (50, 67), (50, 68), (50, 69), 
                                          (50, 70), (50, 71), (51, 67), (51, 68), (51, 69), (51, 70), (51, 71)]
    

    
        for waypoint in rearranged_waypoints:
            s_goal = waypoint  # Discretize the waypoint

            path = Astar.A_s(s_start, s_goal, all_discretized_obstacles)
            path = path[::-1]
            paths.append(path)
            s_start = s_goal  # Update the starting point for the next iteration

        concatenated_path = []

        for path in paths:
            if not concatenated_path:
                concatenated_path.extend(path)
            else:
                concatenated_path.extend(path[1:])

        
        # path = path[::-1]
        for coord in concatenated_path:
            x, y = coord
            waypoints.append((x, y, 1))

        # Polynomial fit.
        self.waypoints = np.array(waypoints)
        deg =25
        t = np.arange(self.waypoints.shape[0])
        fx = np.poly1d(np.polyfit(t, self.waypoints[:,0], deg))
        fy = np.poly1d(np.polyfit(t, self.waypoints[:,1], deg))
        fz = np.poly1d(np.polyfit(t, self.waypoints[:,2], deg))
        duration =8
        t_scaled = np.linspace(t[0], t[-1], int(duration*self.CTRL_FREQ))
        self.ref_x = fx(t_scaled)
        self.ref_y = fy(t_scaled)
        self.ref_z = fz(t_scaled)
        #########################
        # REPLACE THIS (END) ####
        #########################
        t2 = time.perf_counter()
        print("Planning time: ", t2-t1)
        return t_scaled

    def cmdFirmware(self,
                    time,
                    obs,
                    reward=None,
                    done=None,
                    info=None
                    ):
        """Pick command sent to the quadrotor through a Crazyswarm/Crazyradio-like interface.

        INSTRUCTIONS:
            Re-implement this method to return the target position, velocity, acceleration, attitude, and attitude rates to be sent
            from Crazyswarm to the Crazyflie using, e.g., a `cmdFullState` call.

        Args:
            time (float): Episode's elapsed time, in seconds.
            obs (ndarray): The quadrotor's Vicon data [x, 0, y, 0, z, 0, phi, theta, psi, 0, 0, 0].
            reward (float, optional): The reward signal.
            done (bool, optional): Wether the episode has terminated.
            info (dict, optional): Current step information as a dictionary with keys
                'constraint_violation', 'current_target_gate_pos', etc.

        Returns:
            Command: selected type of command (takeOff, cmdFullState, etc., see Enum-like class `Command`).
            List: arguments for the type of command (see comments in class `Command`)

        """
        if self.ctrl is not None:
            raise RuntimeError("[ERROR] Using method 'cmdFirmware' but Controller was created with 'use_firmware' = False.")
        
        # [INSTRUCTIONS] 
        # self.CTRL_FREQ is 30 (set in the getting_started.yaml file) 
        # control input iteration indicates the number of control inputs sent to the quadrotor
        iteration = int(time*self.CTRL_FREQ)
        # print("iteration: ", iteration)
        #########################
        # REPLACE THIS (START) ##
        ########################        #
        dt=15
        # print("The info. of the gates ")
        # print(self.NOMINAL_GATES)

        if iteration == 0:
            height = 1
            duration = 2

            command_type = Command(2)  # Take-off.
            args = [height, duration]

        # [INSTRUCTIONS] Example code for using cmdFullState interface   
        elif iteration >= 3*self.CTRL_FREQ and iteration < dt*self.CTRL_FREQ:
            step = min(iteration-3*self.CTRL_FREQ, len(self.ref_x) -1)
            target_pos = np.array([self.ref_x[step], self.ref_y[step], self.ref_z[step]])
            target_vel = np.zeros(3)
            target_acc = np.zeros(3)
            target_yaw = 0.
            target_rpy_rates = np.zeros(3)

            command_type = Command(1)  # cmdFullState.
            args = [target_pos, target_vel, target_acc, target_yaw, target_rpy_rates]

        elif iteration == dt*self.CTRL_FREQ:
            command_type = Command(6)  # Notify setpoint stop.
            args = []

       # [INSTRUCTIONS] Example code for using goTo interface 
        # elif iteration == (dt+1)*self.CTRL_FREQ+1:7
        #     y = self.initial_obs[2]
        #     z = 1.5
        #     yaw = 0.
        #     duration = 6

        #     command_type = Command(5)  # goTo.
        #     args = [[x, y, z], yaw, duration, False]

        elif iteration == (dt+2)*self.CTRL_FREQ:
            height = 0.
            duration = 3

            command_type = Command(3)  # Land.
            args = [height, duration]

        elif iteration == dt*self.CTRL_FREQ-1:
            command_type = Command(4)  # STOP command to be sent once the trajectory is completed.
            args = []

        else:
            command_type = Command(0)  # None.
            args = []

        #########################Example
        # REPLACE THIS (END) ####
        #########################
        # t3 = t.perf_counter()

        # # print( "Planning Time is: ",t3-t1)
        # elapsed_time = t3 - t1
        # print("Elapsed time:", elapsed_time)
        return command_type, args

    def cmdSimOnly(self,
                   time,
                   obs,
                   reward=None,
                   done=None,
                   info=None
                   ):
        """PID per-propeller thrusts with a simplified, software-only PID quadrotor controller.

        INSTRUCTIONS:
            You do NOT need to re-implement this method for the project.
            Only re-implement this method when `use_firmware` == False to return the target position and velocity.

        Args:
            time (float): Episode's elapsed time, in seconds.
            obs (ndarray): The quadrotor's state [x, x_dot, y, y_dot, z, z_dot, phi, theta, psi, p, q, r].
            reward (float, optional): The reward signal.
            done (bool, optional): Wether the episode has terminated.
            info (dict, optional): Current step information as a dictionary with keys
                'constraint_violation', 'current_target_gate_pos', etc.

        Returns:
            List: target position (len == 3).
            List: target velocity (len == 3).

        """
        if self.ctrl is None:
            raise RuntimeError("[ERROR] Attempting to use method 'cmdSimOnly' but Controller was created with 'use_firmware' = True.")

        iteration = int(time*self.CTRL_FREQ)

        #########################
        if iteration < len(self.ref_x):
            target_p = np.array([self.ref_x[iteration], self.ref_y[iteration], self.ref_z[iteration]])
        else:
            target_p = np.array([self.ref_x[-1], self.ref_y[-1], self.ref_z[-1]])
        target_v = np.zeros(3)
        #########################

        return target_p, target_v

    def reset(self):
        """Initialize/reset data buffers and counters.

        Called once in __init__().

        """
        # Data buffers.
        self.action_buffer = deque([], maxlen=self.BUFFER_SIZE)
        self.obs_buffer = deque([], maxlen=self.BUFFER_SIZE)
        self.reward_buffer = deque([], maxlen=self.BUFFER_SIZE)
        self.done_buffer = deque([], maxlen=self.BUFFER_SIZE)
        self.info_buffer = deque([], maxlen=self.BUFFER_SIZE)

        # Counters.
        self.interstep_counter = 0
        self.interepisode_counter = 0

    # NOTE: this function is not used in the course project. 
    def interEpisodeReset(self):
        """Initialize/reset learning timing variables.

        Called between episodes in `getting_started.py`.

        """
        # Timing stats variables.
        self.interstep_learning_time = 0
        self.interstep_learning_occurrences = 0
        self.interepisode_learning_time = 0
