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
import numpy as np
import time

from collections import deque

try:
    from project_utils import Command, PIDController, timing_step, timing_ep, plot_trajectory, draw_trajectory
except ImportError:
    # PyTest import.
    from .project_utils import Command, PIDController, timing_step, timing_ep, plot_trajectory, draw_trajectory


#########################
# REPLACE THIS (START) ##
#########################
from scipy.interpolate import make_interp_spline

# Optionally, create and import modules you wrote.
# Please refrain from importing large or unstable 3rd party packages.
try:
    from scipy.interpolate import make_interp_spline
    from QuadTreeMap import *
    import example_custom_utils as ecu
except ImportError:
    # PyTest import.
    from . import example_custom_utils as ecu

DT = 10# Duration Constant, adjusts speed of the quadrotor
#########################
# REPLACE THIS (END) ####
#########################

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


    def planning(self, use_firmware, initial_info):
        """Trajectory planning algorithm"""
        #########################
        # REPLACE THIS (START) ##
        #########################
        # t1 - time before environment build
        t1 = time.perf_counter()

        ############### Environment Modeling ################
        center = np.array([0., 0., 1.]) # Center of Environment
        size = np.array([6, 6, 2])      # Dimensions of Environment
        Tree = quadtreeNode(center, size)  # Initialize Root of QuadTree (Class defined in QuadTreeMap.py)
        start = np.array([-1., -3., 1.])    # Start Point
        pt_s =  np.array([-1., -3., 1.])
        Goal = Point([-0.5, 2.0, 1.0], 0.025) # Goal Point after passing all four gates

        ######### Insert environment objects #############
        # Insert Goal Position, giving it a Gate flag as
        Tree.insert(Goal, 'Gate')
        # Load Gate Info to create a list of Gate class instances (Class defined in QuadTreeMap.py)
        G = [Gate(self.NOMINAL_GATES[i]) for i in range(len(self.NOMINAL_GATES))]
        # Insert each Gate into tree (recursively refines tree until max depth is reached)
        for i in range(len(G)):
            Tree.insert(G[i], flag='Gate')
            #G[i].draw()

        # Load obstacle info to create a list of obstacle class instances - (Class defined in QuadTreeMap.py)
        # added buffer into the size as 20 cm square instead of a 12 cm diameter
        O = [Obs(self.NOMINAL_OBSTACLES[i], 0.2, 0.2) for i in range(len(self.NOMINAL_OBSTACLES))]
        # Add to obstacle list the virtual obstacles generated by the edge of the gates
        for i in range(len(self.NOMINAL_GATES)):
            for j in range(len(G[i].obs)):
                O.append(G[i].obs[j])
        # Insert Obstacles in to Tree (recursively refines tree until max depth is reached)
        for i in range(len(O)):
            Tree.insert(O[i], flag='Obs')
            #O[i].draw()

        ## T2 end of Environmment Build and Begin of Motion plan
        t2 = time.perf_counter()

        ####################### Motion Planning ##########################
        GateSequence = [0,1,2,3]
        count = 0
        ini = start
        for k in GateSequence:
            # find waypoints from start to k gate
            # awps contains center points of each node along path
            awps = Astar(Tree, ini, G[k], 1)

            # figure out the delta to reach the other side of the gate
            # considering the orientation of gate
            if G[k].isRotated: # if gate is rotated
                if awps[-1][0] < G[k].cx:   # if the last waypoint is on the LHS of the gate
                    delta = [0.5, 0., 0.]
                else:   # if the last waypoint is on the RHS of the gate
                    delta = [-0.5, 0., 0.]
            else:   # if the last waypoint is  under the gate
                if awps[-1][1] < G[k].cy:
                    delta = [0.0, 0.5, 0.]
                else:   # if the last waypoint is above the gate
                    delta = [0.0, -0.5, 0.]

            if count == 0: # first gate initializations of stacked waypoints using start
                AWPS = np.vstack((start, awps))
            else:           # stack latest waypoints onto total waypoints
                AWPS = np.vstack((AWPS, awps))
            count +=1
            # update start point to other side of gate
            ini = awps[-1] + delta

        ## NOT USED - Find path from last gate position (start)
        ## awps = Astar(Tree, start, Goal, 1)
        ## AWPS = np.vstack((AWPS, awps))

        # Add on final point after the gate (added multiple times to ensure
        # the moving average filter pushes the averaged point to the other side of the gate
        AWPS = np.vstack((AWPS, ini,ini,ini,ini))

        # Refine the coarse WPs using a moving average filter (Function defined in QuadTreeMap.py)
        AWPS_smooth = moving_avg_filter(AWPS, 3)
        # Add Z = 1 to all waypoints generated by stacking a vector of ones
        #AWPS_smooth = np.hstack((AWPS_smooth, np.ones((len(AWPS_smooth), 1))))
        # During the moving_avg, the start point gets lost due to averaging
        AWPS_smooth = np.vstack((start, AWPS_smooth))

        #################### Rescaling #########################
        t = np.arange(AWPS_smooth.shape[0])
        # Define the x and y values for the spline interpolation
        duration = DT
        t_scaled = np.linspace(t[0], t[-1], int(duration * self.CTRL_FREQ))
        splx = make_interp_spline(t, AWPS_smooth[:, 0])
        sply = make_interp_spline(t, AWPS_smooth[:, 1])
        # Evaluate the spline at the new x values
        APS_xnew = splx(t_scaled)
        APS_ynew = sply(t_scaled)

        self.waypoints = AWPS
        self.ref_x = APS_xnew
        self.ref_y = APS_ynew
        self.ref_z = np.ones(len(t_scaled))

        # t3 time for end of motion planning
        t3 = time.perf_counter()
        print(f'Gate Sequence: {GateSequence} \t Number of Waypoints; {len(AWPS)}, Planning Time: {t3-t2:0.4f}, Env Time: {t2-t1:0.4f}')

        #########################
        # REPLACE THIS (END) ####
        #########################

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

        #########################
        # REPLACE THIS (START) ##
        #########################

        # print("The info. of the gates ")
        # print(self.NOMINAL_GATES)
        dt = DT
        if iteration == 0:
            height = 1
            duration = 2

            command_type = Command(2)  # Take-off.
            args = [height, duration]

        # [INSTRUCTIONS] Example code for using cmdFullState interface   
        elif iteration >= 2*self.CTRL_FREQ and iteration <dt*self.CTRL_FREQ:
            step = min(iteration-2*self.CTRL_FREQ, len(self.ref_x) -1)
            target_pos = np.array([self.ref_x[step], self.ref_y[step], self.ref_z[step]])
            target_vel = np.zeros(3)
            target_acc = np.zeros(3)
            target_yaw = 0.
            target_rpy_rates = np.zeros(3)

            command_type = Command(1)  # cmdFullState.
            args = [target_pos, target_vel, target_acc, target_yaw, target_rpy_rates]

        elif iteration == dt * self.CTRL_FREQ:
            command_type = Command(6)  # Notify setpoint stop.
            args = []

       # [INSTRUCTIONS] Example code for using goTo interface
        elif iteration == dt*self.CTRL_FREQ+1:
            x = self.ref_x[-1]
            y = self.ref_y[-1]
            z = 1.0
            yaw = 0.
            duration = 2.5

            command_type = Command(5)  # goTo.
            args = [[x, y, z], yaw, duration, False]
            print('landing')

        elif iteration == (dt+2)*self.CTRL_FREQ:
            height = 0.
            duration = 3

            command_type = Command(3)  # Land.
            args = [height, duration]


        else:
            command_type = Command(0)  # None.
            args = []

        #########################
        # REPLACE THIS (END) ####
        #########################
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
