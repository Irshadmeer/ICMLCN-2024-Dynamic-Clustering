import numpy as np
#import gym
import gymnasium as gym
from gym import spaces

from reliability import calculate_outage_probabilities
from movement import step_sde

class BasicScenario(gym.Env):
    def __init__(self, num_users, active_num_users,num_base_stations,
                 max_coordinates, max_power_db,
                 reliability_constraint, freq, rec_sensitivity,
                 bs_locations,
                 num_antennas,
                 max_time=1000):
        self.active_num_users = active_num_users
        self.num_users = num_users  # Number of total users which can be serviced
        self.num_base_stations = num_base_stations  # Number of base stations
        self.max_coordinates = max_coordinates  # Maximum values in (x, y, z) directions
        self.max_power_db = max_power_db
        self.reliability_constraint = reliability_constraint
        self.bs_locations = bs_locations
        self.freq = freq
        self.rec_sensitivity = 10**(rec_sensitivity/10)
        self.num_antennas = num_antennas

        if bs_locations is None:
            bs_locations = np.random.rand(self.num_base_stations, 3)*self.max_coordinates
        assert np.shape(bs_locations) == (num_base_stations, 3)
        self.bs_locations = bs_locations
        self.active_clusters = np.zeros((num_users, self.num_base_stations), dtype=int)

        

        self.observation_space = spaces.Dict(
                {
                 "user_locations": spaces.Box(low=0, high=np.inf, shape=(num_users, 3), dtype=float),
                 "user_velocities": spaces.Box(low=-np.inf, high=np.inf, shape=(num_users, 3), dtype=float),
                 "los": spaces.MultiBinary([num_users, num_base_stations]),
                  "status": spaces.MultiBinary(num_users, )
                })
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(num_users, num_base_stations), dtype=np.float32
        )

        self.state = {k: None for k in self.observation_space}
        self.time_slot = 0
        self.max_time = max_time

        self.movement_state = [{"x": [0, 0, 0],
                                "y": [0, 0, 0]}
                               for _ in range(self.num_users)]

    def reset(self):
        self.time_slot = 0
        self.movement_state = [{"x": [0, 0, 0],
                                "y": [0, 0, 0]}
                               for _ in range(self.num_users)]
        initial_locations = np.zeros((self.num_users, 3))
        initial_locations[:, :2] = np.random.uniform(0, self.max_coordinates[:2], size=(self.num_users, 2))
        initial_locations[:, 2] = (.5+.5*np.random.rand(self.num_users))*self.max_coordinates[2] #z component 
        self.initial_locations = initial_locations
        user_locations = initial_locations
        user_velocities = np.zeros((self.num_users, 3))  # Velocity vector (x, y, z)
        los = np.random.randint(0, 2, size=(self.num_users, self.num_base_stations))
        status = np.zeros((self.num_users, ))
        random_indices = np.random.choice(self.num_users, self.active_num_users, replace=False) #deciding which ones are active
        status[random_indices]=1 
        state_space = {
                "user_locations": user_locations,
                "user_velocities": user_velocities,
                "los": los,
                "status": status,
                }
        self.state = state_space
        self.info = {"time_slot": self.time_slot,
                "bs_locations": self.bs_locations,
                "user_locations": user_locations,             
               } # basic information
        return state_space
   
    def _get_new_state(self):
        old_locations = self.state['user_locations']
        new_locations = np.zeros_like(old_locations)
        new_locations[:, 2] = old_locations[:, 2]  # keep z component
        status =self.state['status']
        
        scale = 250
        for user in range(self.num_users):
            if status[user] == 1:
                _current_state_x = self.movement_state[user]["x"]
                _current_state_y = self.movement_state[user]["y"]
                _new_state_x = step_sde(_current_state_x, self.time_slot, dt=10/scale)
                _new_state_y = step_sde(_current_state_y, self.time_slot, dt=10/scale)
                self.movement_state[user]["x"] = _new_state_x
                self.movement_state[user]["y"] = _new_state_y
                new_locations[user, 0] = scale*_new_state_x[0] + self.initial_locations[user, 0]
                new_locations[user, 1] = scale*_new_state_y[0] + self.initial_locations[user, 1]
        new_velocities = np.abs(new_locations[:, :3] - old_locations[:, :3])
        new_locations = np.array(new_locations, dtype=float)
       

        los_change = np.random.choice([0, 1], size=(self.num_users, self.num_base_stations), p=[.9, .1])
        los = np.mod(self.state["los"] + los_change, 2)
        new_state = {
                     "user_locations": new_locations,
                     "user_velocities": new_velocities,
                     "los": los,
                     "status": status,
                     }
        self.time_slot = self.time_slot + 1
        return new_state

    def _get_outage_constraint(self):
        user_locations = self.state["user_locations"]
        status=self.state["status"]
        user_locations = user_locations[status==1]
        _users_in_box_x = np.logical_and(self.max_coordinates[0]/2 < user_locations[:, 0],
                                         user_locations[:, 0] < 2*self.max_coordinates[0]/3)
        _users_in_box_y = np.logical_and(self.max_coordinates[1]/2 < user_locations[:, 1],
                                         user_locations[:, 1] < 2*self.max_coordinates[1]/3)
        _users_in_box = np.logical_and(_users_in_box_x, _users_in_box_y)
        outage_constraints = np.where(_users_in_box,
                                      self.reliability_constraint/100,                            # change here for the varying reliability constraints.
                                      self.reliability_constraint)
        return outage_constraints, _users_in_box
    
    def _count_handover_user(self, current_clusters, status):
        old_clusters = self.active_clusters[status == 1]
        current_clusters =current_clusters[status==1]
        changes_per_user = np.any(old_clusters != current_clusters, axis=1)
        return changes_per_user

    
    
    
    def step(self, action):

        ###################
        user_locations = self.state["user_locations"]
        status =self.state["status"]
        num_active_users = np.count_nonzero(status == 1)

        active_user_action = action[status == 1]

        filtered_action = action * status[:, np.newaxis]
        filtered_action[status != 1] = 0

        base_station_powers = 10**(self.max_power_db*active_user_action/10.)  # convert dB power (action) in linear scale
        los = self.state['los']
        
        comp_power = np.ones((num_active_users,self.num_base_stations))
        max_power = 10**(self.max_power_db*comp_power/10.)

        outage_probabilities = calculate_outage_probabilities(
                user_locations[status == 1], self.bs_locations, base_station_powers, 
                self.freq, self.rec_sensitivity,status, los[status == 1], self.num_antennas)

        total_power_per_user =np.sum(base_station_powers, axis=1)
        total_power = np.sum(base_station_powers)
        power_fraction = total_power/np.sum(max_power)
       
        active_num_bs = np.zeros(self.num_users,)
        active_bs_indices = np.empty(len(active_user_action), dtype=object)
        current_cluster = np.zeros((self.num_users, self.num_base_stations), dtype=int)

    
        for j,row in enumerate(filtered_action):
           current_cluster[j] = np.where(row > 0, 1, 0)
       
        for i,row in enumerate(active_user_action):
           active_num_bs[i] = len([x for x in row if x > 0])
           active_bs_indices[i] = [j for j, x in enumerate(row) if x > 0]
           

       
        
        #handovers
        handovers=self._count_handover_user(current_clusters=current_cluster,status=status)
        self.active_clusters=current_cluster
        num_active_connections =np.sum(active_num_bs)
        eps_max,users_in_box = self._get_outage_constraint()
        temp = eps_max-outage_probabilities
        outage_violations = np.count_nonzero(outage_probabilities > eps_max)
       
        reward_reliability = len([x for x in temp if x < 0])/num_active_users
        reward_active_bs   = 1-(num_active_connections/(num_active_users*self.num_base_stations))
        reward_ee          = 1-total_power/np.sum(max_power)
        handover_rate      = sum(handovers)/num_active_users
        reward_handover    = 1-handover_rate 
        



        reward = reward_ee  + reward_active_bs + reward_handover - reward_reliability 
        
       
        info = {"time_slot": self.time_slot,
                "action": action,
                "active_user_action":active_user_action,
                "total_power_per_user":total_power_per_user,
                "total_power": total_power,
                "energy_efficiency": 1./total_power,
                "reward": reward,
                "reward_ee": reward_ee,
                "reward_reliability": reward_reliability,
                "outage_probabilities": outage_probabilities,
                "outage_violations": outage_violations,
                "power_fraction": power_fraction,
                "num_active_BS": active_num_bs,
                "bs_locations": self.bs_locations,
                "user_locations": user_locations,
                "active_bs_index": active_bs_indices,
                "user_in_high_rel_zone":users_in_box,
                "active_num_users":num_active_users,
                "handovers":handovers,
                "handover_rate":handover_rate,
                "reward_handover":reward_handover}
        
        
        info.update(self.state)
        self.info = info
        new_state = self._get_new_state()
        new_user_locations = new_state['user_locations']
        
        #user dropping
       
        dropped_users = np.where((status.reshape(-1) == 1) & ((np.any(new_user_locations > self.max_coordinates, axis=1)) | (np.any(new_user_locations < 0, axis=1))))[0]

        #user adding

        if self.time_slot==(round(self.max_time/4)|round(self.max_time/3)|round(self.max_time/2)):
            added_users = np.random.choice(np.where(status == 0)[0]) if np.any(status == 0) else None
            status[added_users] = 1

        if dropped_users.size > 0:
            status[dropped_users] = 0
        new_state["status"] = status




        self.state = new_state

        done = self.is_done()
        return new_state, reward, done, info

    def is_done(self):
        if self.time_slot > self.max_time:
           return True
        status = self.state['status']
        if np.sum(status)<=1:
            return True
        return False


if __name__ == "__main__":
    from stable_baselines3.common.env_checker import check_env

    num_uavs = 1
    num_bs = 4
    max_x = 500
    max_y = 500
    max_z = 50
    max_power_db = 20
    reliability_constraint = .1
    carrier_freq = 2.4e9  # Hz
    rec_sensitivity = -100  # dBm

    env = BasicScenario(num_uavs, num_bs, [max_x, max_y, max_z], max_power_db,
                        reliability_constraint, carrier_freq, rec_sensitivity)
    # It will check your custom environment and output additional warnings if needed
    check_env(env)
