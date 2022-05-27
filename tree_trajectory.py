import random
import tensorflow as tf
import numpy as np

class TreeTrajectoryDataset(tf.data.Dataset):
    
    def __new__(cls, num_trajectorys=3):
      return tf.data.Dataset.from_generator(cls._generator, output_types=(tf.dtypes.float32, tf.dtypes.int32), args=(num_trajectorys,))
    
    def __init__(self, tree_data) -> None:
        super().__init__()
        self.tree_data=tree_data
    
    def _generator(self, num_trajectorys):
        while True:
            trajectory_names = self.tree_data.get_trajectory_names()
            #print("len(trajectory_names): ", len(trajectory_names))
            
            #https://minerl.io/docs/api/data.html
            trajectory_name = random.choice(trajectory_names)
            print("trajectory_name: ", trajectory_name)
            
            trajectory = self.tree_data.load_data(trajectory_name, skip_interval=0, include_metadata=False)
            #print("trajectory: ", trajectory)
            
            noop_action_num = 0
            
            all_actions = []
            all_obs = []
            for dataset_observation, dataset_action, reward, next_state, done in trajectory:  
                #print("reward: ", reward)
                
                #state_pov = dataset_observation['pov']
                #observation = np.concatenate((dataset_observation['pov'] / 255.0, inventory_channel), axis=2)
                # OrderedDict([('pov', array([[[ 0,  0,  0],
                #         [ 0,  0,  2],
                #         [ 0,  2,  0],
                #         ...,
                #         [30, 57, 16],
                #         [ 0,  2,  0],
                #         [ 0,  2,  0]]], dtype=uint8))])
                observation = dataset_observation['pov'] / 255.0

                #OrderedDict([('attack', 1), ('back', 0), ('camera', array([0., 0.], dtype=float32)), ('forward', 0), ('jump', 0), ('left', 0), ('right', 0), ('sneak', 0), ('sprint', 0)])
                action_camera_0 = dataset_action['camera'][0]
                action_camera_1 = dataset_action['camera'][1]
                action_attack = dataset_action['attack']
                action_forward = dataset_action['forward']
                action_jump = dataset_action['jump']
                action_back = dataset_action['back']
                action_left = dataset_action['left']
                action_right = dataset_action['right']
                action_sneak = dataset_action['sneak']

                camera_threshols = (abs(action_camera_0) + abs(action_camera_1)) / 2.0
                if (camera_threshols > 2.5):
                    if ( (action_camera_1 < 0) & ( abs(action_camera_0) < abs(action_camera_1) ) ):
                        if (action_attack == 1):
                            action_index = 0
                        elif (action_forward == 1):
                            action_index = 1
                        elif (action_left == 1):
                            action_index = 2
                        elif (action_right == 1):
                            action_index = 3
                        elif (action_back == 1):
                            action_index = 4
                        elif (action_jump == 1):
                            action_index = 5
                        else:
                            action_index = 6
                    elif ( (action_camera_1 > 0) & ( abs(action_camera_0) < abs(action_camera_1) ) ):
                        if (action_attack == 1):
                            action_index = 7
                        elif (action_forward == 1):
                            action_index = 8
                        elif (action_left == 1):
                            action_index = 9
                        elif (action_right == 1):
                            action_index = 10
                        elif (action_back == 1):
                            action_index = 11
                        elif (action_jump == 1):
                            action_index = 12
                        else:
                            action_index = 13
                    elif ( (action_camera_0 < 0) & ( abs(action_camera_0) > abs(action_camera_1) ) ):
                        if (action_attack == 1):
                            action_index = 14
                        elif (action_forward == 1):
                            action_index = 15
                        elif (action_left == 1):
                            action_index = 16
                        elif (action_right == 1):
                            action_index = 17
                        elif (action_back == 1):
                            action_index = 18
                        elif (action_jump == 1):
                            action_index = 19
                        else:
                            action_index = 20
                    elif ( (action_camera_0 > 0) & ( abs(action_camera_0) > abs(action_camera_1) ) ):
                        if (action_attack == 1):
                            action_index = 21
                        elif (action_forward == 1):
                            action_index = 22
                        elif (action_left == 1):
                            action_index = 23
                        elif (action_right == 1):
                            action_index = 24
                        elif (action_back == 1):
                            action_index = 25
                        elif (action_jump == 1):
                            action_index = 26
                        else:
                            action_index = 27

                elif (action_forward == 1):
                    if (action_attack == 1):
                        action_index = 28
                    elif (action_jump == 1):
                        action_index = 29
                    else:
                        action_index = 30
                elif (action_jump == 1):
                    if (action_attack == 1):
                        action_index = 31
                    else:
                        action_index = 32
                elif (action_back == 1):
                    if (action_attack == 1):
                        action_index = 33
                    else:
                        action_index = 34
                elif (action_left == 1):
                    if (action_attack == 1):
                        action_index = 35
                    else:
                        action_index = 36
                elif (action_right == 1):
                    if (action_attack == 1):
                        action_index = 37
                    else:
                        action_index = 38
                elif (action_sneak == 1):
                    if (action_attack == 1):
                        action_index = 39
                    else:
                        action_index = 40
                elif (action_attack == 1):
                    action_index = 41
                else:
                    action_index = 42

                if (dataset_action['attack'] == 0 and dataset_action['back'] == 0 and dataset_action['camera'][0] == 0.0 and 
                    dataset_action['camera'][1] == 0.0 and dataset_action['forward'] == 0 and dataset_action['jump'] == 0 and 
                    dataset_action['left'] == 0 and dataset_action['right'] == 0 and dataset_action['sneak'] == 0):
                    #print("continue: ")
                    continue

                if action_index == 41:
                    #print("camera_threshols: ", camera_threshols)
                    #print("dataset_action: ", dataset_action)
                    noop_action_num += 1
                    
                #print("observation.shape: ", observation.shap
                #print("action_index: ", action_index)
                #print("done: ", done)

                all_obs.append(observation)
                all_actions.append(np.array([action_index]))

            print("len(all_obs): ", len(all_obs))
            print("noop_action_num: ", noop_action_num)
            print("")
            yield (all_obs, all_actions)

            break
    
    