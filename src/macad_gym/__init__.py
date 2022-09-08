import os
import sys
import logging

from gym.envs.registration import register

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "yes please"
LOG_DIR = os.path.join(os.getcwd(), "logs")
if not os.path.isdir(LOG_DIR):
    os.mkdir(LOG_DIR)

# Init and setup the root logger
logging.basicConfig(filename=LOG_DIR + '/macad-gym.log', level=logging.DEBUG)

# Fix path issues with included CARLA API
sys.path.append(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "carla/PythonAPI"))

# Declare available environments with a brief description
_AVAILABLE_ENVS = {
    'HomoNcomIndePOIntrxMASS3CTWN3-v0': {
        "entry_point":
        "macad_gym.envs:HomoNcomIndePOIntrxMASS3CTWN3",
        "description":
        "Homogeneous, Non-communicating, Independent, Partially-"
        "Observable Intersection Multi-Agent scenario with "
        "Stop-Sign, 3 Cars in Town3, version 0"
    },
    'UrbanScenario2A3C-v0': {
        "entry_point":
        "macad_gym.envs:UrbanScenario2A3C",
        "description":
        "Heterogeneous, Non-communicating, Independent,"
        "Partially-Observable Intersection Multi-Agent"
        " scenario with Traffic-Light Signal, 1-Bike, 2-Car,"
        "1-Pedestrian in Town3, version 0"
    },
    'UrbanScenario2IMPALA-v0': {
        "entry_point":
        "macad_gym.envs:UrbanScenario2IMPALA",
        "description":
        "Heterogeneous, Non-communicating, Independent,"
        "Partially-Observable Intersection Multi-Agent"
        " scenario with Traffic-Light Signal, 1-Bike, 2-Car,"
        "1-Pedestrian in Town3, version 0"
    },
# Below the line are QRS based environments

    'Experimental-v0': {
        "entry_point":
        "macad_gym.envs:Experimental",
        "description":
        "Homogeneous, Non-communicating, Independent, Partially-"
        "Observable Intersection Multi-Agent scenario with "
        "Stop-Sign, 3 Cars in Town3, version 0"
    },


    'PPO_straight_train-v0': {
        "entry_point":
        "macad_gym.envs:PPO_straight_train",
        "description":
        "Homogeneous, Non-communicating, Independent, Partially-"
        "Observable Intersection Multi-Agent scenario with "
        "Stop-Sign, 3 Cars in Town3, version 0"
    },
    'PPO_three_way_train-v0': {
        "entry_point":
        "macad_gym.envs:PPO_three_way_train",
        "description":
        "Homogeneous, Non-communicating, Independent, Partially-"
        "Observable Intersection Multi-Agent scenario with "
        "Stop-Sign, 3 Cars in Town3, version 0"
    },
    'PPO_four_way_train-v0': {
        "entry_point":
        "macad_gym.envs:PPO_four_way_train",
        "description":
        "Homogeneous, Non-communicating, Independent, Partially-"
        "Observable Intersection Multi-Agent scenario with "
        "Stop-Sign, 3 Cars in Town3, version 0"
    },
    'PPO_roundabout_train-v0': {
        "entry_point":
        "macad_gym.envs:PPO_roundabout_train",
        "description":
        "Homogeneous, Non-communicating, Independent, Partially-"
        "Observable Intersection Multi-Agent scenario with "
        "Stop-Sign, 3 Cars in Town3, version 0"
    },
    'PPO_merge_train-v0': {
        "entry_point":
        "macad_gym.envs:PPO_merge_train",
        "description":
        "Homogeneous, Non-communicating, Independent, Partially-"
        "Observable Intersection Multi-Agent scenario with "
        "Stop-Sign, 3 Cars in Town3, version 0"
    },


    'A2C_straight_train-v0': {
        "entry_point":
        "macad_gym.envs:A2C_straight_train",
        "description":
        "Homogeneous, Non-communicating, Independent, Partially-"
        "Observable Intersection Multi-Agent scenario with "
        "Stop-Sign, 3 Cars in Town3, version 0"
    },
    'A2C_three_way_train-v0': {
        "entry_point":
        "macad_gym.envs:A2C_three_way_train",
        "description":
        "Homogeneous, Non-communicating, Independent, Partially-"
        "Observable Intersection Multi-Agent scenario with "
        "Stop-Sign, 3 Cars in Town3, version 0"
    },
    'A2C_four_way_train-v0': {
        "entry_point":
        "macad_gym.envs:A2C_four_way_train",
        "description":
        "Homogeneous, Non-communicating, Independent, Partially-"
        "Observable Intersection Multi-Agent scenario with "
        "Stop-Sign, 3 Cars in Town3, version 0"
    },
    'A2C_roundabout_train-v0': {
        "entry_point":
        "macad_gym.envs:A2C_roundabout_train",
        "description":
        "Homogeneous, Non-communicating, Independent, Partially-"
        "Observable Intersection Multi-Agent scenario with "
        "Stop-Sign, 3 Cars in Town3, version 0"
    },
    'A2C_merge_train-v0': {
        "entry_point":
        "macad_gym.envs:A2C_merge_train",
        "description":
        "Homogeneous, Non-communicating, Independent, Partially-"
        "Observable Intersection Multi-Agent scenario with "
        "Stop-Sign, 3 Cars in Town3, version 0"
    },


    'A3C_straight_train-v0': {
        "entry_point":
        "macad_gym.envs:A3C_straight_train",
        "description":
        "Homogeneous, Non-communicating, Independent, Partially-"
        "Observable Intersection Multi-Agent scenario with "
        "Stop-Sign, 3 Cars in Town3, version 0"
    },
    'A3C_three_way_train-v0': {
        "entry_point":
        "macad_gym.envs:A3C_three_way_train",
        "description":
        "Homogeneous, Non-communicating, Independent, Partially-"
        "Observable Intersection Multi-Agent scenario with "
        "Stop-Sign, 3 Cars in Town3, version 0"
    },
    'A3C_four_way_train-v0': {
        "entry_point":
        "macad_gym.envs:A3C_four_way_train",
        "description":
        "Homogeneous, Non-communicating, Independent, Partially-"
        "Observable Intersection Multi-Agent scenario with "
        "Stop-Sign, 3 Cars in Town3, version 0"
    },
    'A3C_roundabout_train-v0': {
        "entry_point":
        "macad_gym.envs:A3C_roundabout_train",
        "description":
        "Homogeneous, Non-communicating, Independent, Partially-"
        "Observable Intersection Multi-Agent scenario with "
        "Stop-Sign, 3 Cars in Town3, version 0"
    },
    'A3C_merge_train-v0': {
        "entry_point":
        "macad_gym.envs:A3C_merge_train",
        "description":
        "Homogeneous, Non-communicating, Independent, Partially-"
        "Observable Intersection Multi-Agent scenario with "
        "Stop-Sign, 3 Cars in Town3, version 0"
    },

    'IMPALA_straight_train-v0': {
        "entry_point":
        "macad_gym.envs:IMPALA_straight_train",
        "description":
        "Homogeneous, Non-communicating, Independent, Partially-"
        "Observable Intersection Multi-Agent scenario with "
        "Stop-Sign, 3 Cars in Town3, version 0"
    },
    'IMPALA_three_way_train-v0': {
        "entry_point":
        "macad_gym.envs:IMPALA_three_way_train",
        "description":
        "Homogeneous, Non-communicating, Independent, Partially-"
        "Observable Intersection Multi-Agent scenario with "
        "Stop-Sign, 3 Cars in Town3, version 0"
    },
    'IMPALA_four_way_train-v0': {
        "entry_point":
        "macad_gym.envs:IMPALA_four_way_train",
        "description":
        "Homogeneous, Non-communicating, Independent, Partially-"
        "Observable Intersection Multi-Agent scenario with "
        "Stop-Sign, 3 Cars in Town3, version 0"
    },
    'IMPALA_roundabout_train-v0': {
        "entry_point":
        "macad_gym.envs:IMPALA_roundabout_train",
        "description":
        "Homogeneous, Non-communicating, Independent, Partially-"
        "Observable Intersection Multi-Agent scenario with "
        "Stop-Sign, 3 Cars in Town3, version 0"
    },
    'IMPALA_merge_train-v0': {
        "entry_point":
        "macad_gym.envs:IMPALA_merge_train",
        "description":
        "Homogeneous, Non-communicating, Independent, Partially-"
        "Observable Intersection Multi-Agent scenario with "
        "Stop-Sign, 3 Cars in Town3, version 0"
    },



    'DQN_straight_train-v0': {
        "entry_point":
        "macad_gym.envs:DQN_straight_train",
        "description":
        "Homogeneous, Non-communicating, Independent, Partially-"
        "Observable Intersection Multi-Agent scenario with "
        "Stop-Sign, 3 Cars in Town3, version 0"
    },
    'DQN_three_way_train-v0': {
        "entry_point":
        "macad_gym.envs:DQN_three_way_train",
        "description":
        "Homogeneous, Non-communicating, Independent, Partially-"
        "Observable Intersection Multi-Agent scenario with "
        "Stop-Sign, 3 Cars in Town3, version 0"
    },
    'DQN_four_way_train-v0': {
        "entry_point":
        "macad_gym.envs:DQN_four_way_train",
        "description":
        "Homogeneous, Non-communicating, Independent, Partially-"
        "Observable Intersection Multi-Agent scenario with "
        "Stop-Sign, 3 Cars in Town3, version 0"
    },
    'DQN_roundabout_train-v0': {
        "entry_point":
        "macad_gym.envs:DQN_roundabout_train",
        "description":
        "Homogeneous, Non-communicating, Independent, Partially-"
        "Observable Intersection Multi-Agent scenario with "
        "Stop-Sign, 3 Cars in Town3, version 0"
    },
    'DQN_merge_train-v0': {
        "entry_point":
        "macad_gym.envs:DQN_merge_train",
        "description":
        "Homogeneous, Non-communicating, Independent, Partially-"
        "Observable Intersection Multi-Agent scenario with "
        "Stop-Sign, 3 Cars in Town3, version 0"
    },


    'DDPG_straight_train-v0': {
        "entry_point":
        "macad_gym.envs:DDPG_straight_train",
        "description":
        "Homogeneous, Non-communicating, Independent, Partially-"
        "Observable Intersection Multi-Agent scenario with "
        "Stop-Sign, 3 Cars in Town3, version 0"
    },
    'DDPG_three_way_train-v0': {
        "entry_point":
        "macad_gym.envs:DDPG_three_way_train",
        "description":
        "Homogeneous, Non-communicating, Independent, Partially-"
        "Observable Intersection Multi-Agent scenario with "
        "Stop-Sign, 3 Cars in Town3, version 0"
    },
    'DDPG_four_way_train-v0': {
        "entry_point":
        "macad_gym.envs:DDPG_four_way_train",
        "description":
        "Homogeneous, Non-communicating, Independent, Partially-"
        "Observable Intersection Multi-Agent scenario with "
        "Stop-Sign, 3 Cars in Town3, version 0"
    },
    'DDPG_roundabout_train-v0': {
        "entry_point":
        "macad_gym.envs:DDPG_roundabout_train",
        "description":
        "Homogeneous, Non-communicating, Independent, Partially-"
        "Observable Intersection Multi-Agent scenario with "
        "Stop-Sign, 3 Cars in Town3, version 0"
    },
    'DDPG_merge_train-v0': {
        "entry_point":
        "macad_gym.envs:DDPG_merge_train",
        "description":
        "Homogeneous, Non-communicating, Independent, Partially-"
        "Observable Intersection Multi-Agent scenario with "
        "Stop-Sign, 3 Cars in Town3, version 0"
    },


    'TD3_straight_train-v0': {
        "entry_point":
        "macad_gym.envs:TD3_straight_train",
        "description":
        "Homogeneous, Non-communicating, Independent, Partially-"
        "Observable Intersection Multi-Agent scenario with "
        "Stop-Sign, 3 Cars in Town3, version 0"
    },
    'TD3_three_way_train-v0': {
        "entry_point":
        "macad_gym.envs:TD3_three_way_train",
        "description":
        "Homogeneous, Non-communicating, Independent, Partially-"
        "Observable Intersection Multi-Agent scenario with "
        "Stop-Sign, 3 Cars in Town3, version 0"
    },
   
    'TD3_four_way_train-v0': {
        "entry_point":
        "macad_gym.envs:TD3_four_way_train",
        "description":
        "Homogeneous, Non-communicating, Independent, Partially-"
        "Observable Intersection Multi-Agent scenario with "
        "Stop-Sign, 3 Cars in Town3, version 0"
    },
    'TD3_roundabout_train-v0': {
        "entry_point":
        "macad_gym.envs:TD3_roundabout_train",
        "description":
        "Homogeneous, Non-communicating, Independent, Partially-"
        "Observable Intersection Multi-Agent scenario with "
        "Stop-Sign, 3 Cars in Town3, version 0"
    },
    'TD3_merge_train-v0': {
        "entry_point":
        "macad_gym.envs:TD3_merge_train",
        "description":
        "Homogeneous, Non-communicating, Independent, Partially-"
        "Observable Intersection Multi-Agent scenario with "
        "Stop-Sign, 3 Cars in Town3, version 0"
    },



    'PPO_A2C_A3C_straight_train-v0': {
        "entry_point":
        "macad_gym.envs:PPO_A2C_A3C_straight_train",
        "description":
        "Homogeneous, Non-communicating, Independent, Partially-"
        "Observable Intersection Multi-Agent scenario with "
        "Stop-Sign, 3 Cars in Town3, version 0"
    },
    'PPO_A2C_three_way-v0': {
        "entry_point":
        "macad_gym.envs:PPO_A2C_three_way",
        "description":
        "Homogeneous, Non-communicating, Independent, Partially-"
        "Observable Intersection Multi-Agent scenario with "
        "Stop-Sign, 3 Cars in Town3, version 0"
    },
    'A3C_IMPALA_three_way-v0': {
        "entry_point":
        "macad_gym.envs:A3C_IMPALA_three_way",
        "description":
        "Homogeneous, Non-communicating, Independent, Partially-"
        "Observable Intersection Multi-Agent scenario with "
        "Stop-Sign, 3 Cars in Town3, version 0"
    },
    
    'IMPALA_DQN_straight_train-v0': {
        "entry_point":
        "macad_gym.envs:IMPALA_DQN_straight_train",
        "description":
        "Homogeneous, Non-communicating, Independent, Partially-"
        "Observable Intersection Multi-Agent scenario with "
        "Stop-Sign, 3 Cars in Town3, version 0"
    },
    'DDPG_TD3_straight_train-v0': {
        "entry_point":
        "macad_gym.envs:DDPG_TD3_straight_train",
        "description":
        "Homogeneous, Non-communicating, Independent, Partially-"
        "Observable Intersection Multi-Agent scenario with "
        "Stop-Sign, 3 Cars in Town3, version 0"
    },
    'PPO_A2C_A3C_three_way_train-v0': {
        "entry_point":
        "macad_gym.envs:PPO_A2C_A3C_three_way_train",
        "description":
        "Homogeneous, Non-communicating, Independent, Partially-"
        "Observable Intersection Multi-Agent scenario with "
        "Stop-Sign, 3 Cars in Town3, version 0"
    },
    'IMPALA_DQN_three_way-v0': {
        "entry_point":
        "macad_gym.envs:IMPALA_DQN_three_way",
        "description":
        "Homogeneous, Non-communicating, Independent, Partially-"
        "Observable Intersection Multi-Agent scenario with "
        "Stop-Sign, 3 Cars in Town3, version 0"
    },
    'IMPALA_DQN_three_way_train-v0': {
        "entry_point":
        "macad_gym.envs:IMPALA_DQN_three_way_train",
        "description":
        "Homogeneous, Non-communicating, Independent, Partially-"
        "Observable Intersection Multi-Agent scenario with "
        "Stop-Sign, 3 Cars in Town3, version 0"
    },
    'DDPG_TD3_three_way_train-v0': {
        "entry_point":
        "macad_gym.envs:DDPG_TD3_three_way_train",
        "description":
        "Homogeneous, Non-communicating, Independent, Partially-"
        "Observable Intersection Multi-Agent scenario with "
        "Stop-Sign, 3 Cars in Town3, version 0"
    },    
    'DDPG_TD3_three_way-v0': {
        "entry_point":
        "macad_gym.envs:DDPG_TD3_three_way",
        "description":
        "Homogeneous, Non-communicating, Independent, Partially-"
        "Observable Intersection Multi-Agent scenario with "
        "Stop-Sign, 3 Cars in Town3, version 0"
    },
    'PPO_A2C_A3C_four_way_train-v0': {
        "entry_point":
        "macad_gym.envs:PPO_A2C_A3C_four_way_train",
        "description":
        "Homogeneous, Non-communicating, Independent, Partially-"
        "Observable Intersection Multi-Agent scenario with "
        "Stop-Sign, 3 Cars in Town3, version 0"
    },
    'PPO_A2C_four_way-v0': {
        "entry_point":
        "macad_gym.envs:PPO_A2C_four_way",
        "description":
        "Homogeneous, Non-communicating, Independent, Partially-"
        "Observable Intersection Multi-Agent scenario with "
        "Stop-Sign, 3 Cars in Town3, version 0"
    },
    'A3C_IMPALA_four_way-v0': {
        "entry_point":
        "macad_gym.envs:A3C_IMPALA_four_way",
        "description":
        "Homogeneous, Non-communicating, Independent, Partially-"
        "Observable Intersection Multi-Agent scenario with "
        "Stop-Sign, 3 Cars in Town3, version 0"
    },
    'IMPALA_DQN_four_way-v0': {
        "entry_point":
        "macad_gym.envs:IMPALA_DQN_four_way",
        "description":
        "Homogeneous, Non-communicating, Independent, Partially-"
        "Observable Intersection Multi-Agent scenario with "
        "Stop-Sign, 3 Cars in Town3, version 0"
    },
    'IMPALA_DQN_four_way_train-v0': {
        "entry_point":
        "macad_gym.envs:IMPALA_DQN_four_way_train",
        "description":
        "Homogeneous, Non-communicating, Independent, Partially-"
        "Observable Intersection Multi-Agent scenario with "
        "Stop-Sign, 3 Cars in Town3, version 0"
    },
    'DDPG_TD3_four_way_train-v0': {
        "entry_point":
        "macad_gym.envs:DDPG_TD3_four_way_train",
        "description":
        "Homogeneous, Non-communicating, Independent, Partially-"
        "Observable Intersection Multi-Agent scenario with "
        "Stop-Sign, 3 Cars in Town3, version 0"
    },
    'DDPG_TD3_four_way-v0': {
        "entry_point":
        "macad_gym.envs:DDPG_TD3_four_way",
        "description":
        "Homogeneous, Non-communicating, Independent, Partially-"
        "Observable Intersection Multi-Agent scenario with "
        "Stop-Sign, 3 Cars in Town3, version 0"
    },
    'PPO_A2C_A3C_roundabout_train-v0': {
        "entry_point":
        "macad_gym.envs:PPO_A2C_A3C_roundabout_train",
        "description":
        "Homogeneous, Non-communicating, Independent, Partially-"
        "Observable Intersection Multi-Agent scenario with "
        "Stop-Sign, 3 Cars in Town3, version 0"
    },
    'IMPALA_DQN_roundabout_train-v0': {
        "entry_point":
        "macad_gym.envs:IMPALA_DQN_roundabout_train",
        "description":
        "Homogeneous, Non-communicating, Independent, Partially-"
        "Observable Intersection Multi-Agent scenario with "
        "Stop-Sign, 3 Cars in Town3, version 0"
    },
    'DDPG_TD3_roundabout_train-v0': {
        "entry_point":
        "macad_gym.envs:DDPG_TD3_roundabout_train",
        "description":
        "Homogeneous, Non-communicating, Independent, Partially-"
        "Observable Intersection Multi-Agent scenario with "
        "Stop-Sign, 3 Cars in Town3, version 0"
    },
    'PPO_A2C_A3C_merge_train-v0': {
        "entry_point":
        "macad_gym.envs:PPO_A2C_A3C_merge_train",
        "description":
        "Homogeneous, Non-communicating, Independent, Partially-"
        "Observable Intersection Multi-Agent scenario with "
        "Stop-Sign, 3 Cars in Town3, version 0"
    },
    'IMPALA_DQN_merge_train-v0': {
        "entry_point":
        "macad_gym.envs:IMPALA_DQN_merge_train",
        "description":
        "Homogeneous, Non-communicating, Independent, Partially-"
        "Observable Intersection Multi-Agent scenario with "
        "Stop-Sign, 3 Cars in Town3, version 0"
    },
    'DDPG_TD3_merge_train-v0': {
        "entry_point":
        "macad_gym.envs:DDPG_TD3_merge_train",
        "description":
        "Homogeneous, Non-communicating, Independent, Partially-"
        "Observable Intersection Multi-Agent scenario with "
        "Stop-Sign, 3 Cars in Town3, version 0"
    },
    
}
for env_id, val in _AVAILABLE_ENVS.items():
    register(id=env_id, entry_point=val.get("entry_point"))


def list_available_envs():
    print("Environment-ID: Short-description")
    import pprint
    available_envs = {}
    for env_id, val in _AVAILABLE_ENVS.items():
        available_envs[env_id] = val.get("description")
    pprint.pprint(available_envs)
