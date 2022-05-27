import argparse
import random
import tensorflow as tf
import os
import minerl
import tree_trajectory
import network
import numpy as np

def main():
    # do your main minerl code

    #paths
    workspace_path= 'C:/Users/Halim/Downloads/minecraftRL/minecraft_bot_dev-master'
    data_path='C:/Users/Halim/Downloads/minecraftRL/MineRLenv'
    env_name = 'MineRLTreechop'

    #if already trained model and/or gpu exists
    parser = argparse.ArgumentParser(description='Minecraft Supervised Learning')
    parser.add_argument('--pretrained_model', type=str, help='pretrained model name')
    parser.add_argument('--gpu_use', type=bool, default=False, help='use gpu')
    arguments = parser.parse_args()

    #if gpu exists
    if arguments.gpu_use == True:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        tf.config.experimental.set_virtual_device_configuration(gpus[0],
                    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4000)])
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    #location for model summary
    writer = tf.summary.create_file_writer(workspace_path + "/tree_tensorboard")

    
    num_actions = 43
    num_hidden_units = 512

    cce_loss = tf.keras.losses.CategoricalCrossentropy()
    cce_loss_logits = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.Adam(0.0001)

    @tf.function
    def supervised_replay(replay_obs_list, replay_act_list, memory_state, carry_state):
        replay_obs_array = tf.concat(replay_obs_list, 0) #just to convert format to a tf.Tensor
        replay_act_array = tf.concat(replay_act_list, 0)
        replay_memory_state_array = tf.concat(memory_state, 0)
        replay_carry_state_array = tf.concat(carry_state, 0)

        memory_state = replay_memory_state_array
        carry_state = replay_carry_state_array

        batch_size = replay_obs_array.shape[0]
        tf.print("batch_size: ", batch_size)
        
        with tf.GradientTape() as tape:
            act_probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
            for i in tf.range(0, batch_size): #batch size = 32
                prediction = model(tf.expand_dims(replay_obs_array[i,:,:,:], 0), memory_state, carry_state, training=True)
                act_pi = prediction[0]
                memory_state = prediction[2]
                carry_state = prediction[3]
            
                act_probs = act_probs.write(i, act_pi[0])

            act_probs = act_probs.stack()

            tf.print("replay_act_array: ", replay_act_array)
            tf.print("tf.argmax(act_probs, 1): ", tf.argmax(act_probs, 1))

            replay_act_array_onehot = tf.one_hot(replay_act_array, num_actions)
            replay_act_array_onehot = tf.reshape(replay_act_array_onehot, (batch_size, num_actions))
            act_loss = cce_loss_logits(replay_act_array_onehot, act_probs)

            #tf.print("act_loss: ", act_loss)
            regularization_loss = tf.reduce_sum(model.losses)
            total_loss = act_loss + 1e-5 * regularization_loss
        
            #tf.print("total_loss: ", total_loss)
            #tf.print("")
            
        grads = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        return total_loss, memory_state, carry_state


    def supervised_train(dataset, training_episode):
        for batch in dataset:
            episode_size = batch[0].shape[1]
            print("episode_size: ", episode_size)
        
            replay_obs_list = batch[0][0]
            replay_act_list = batch[1][0]
        
            memory_state = np.zeros([1,128], dtype=np.float32)
            carry_state =  np.zeros([1,128], dtype=np.float32)
            step_length = 32
            total_loss = 0
            for episode_index in range(0, episode_size, step_length):
                obs = replay_obs_list[episode_index:episode_index+step_length,:,:,:]
                act = replay_act_list[episode_index:episode_index+step_length,:]
                
                #print("len(obs): ", len(obs))
                if len(obs) != step_length:
                    break
                
                total_loss, next_memory_state, next_carry_state = supervised_replay(obs, act, memory_state, carry_state)
                memory_state = next_memory_state
                carry_state = next_carry_state
            
                print("total_loss: ", total_loss)
                print("")
                
            with writer.as_default():
                tf.summary.scalar("total_loss", total_loss, step=training_episode)
                writer.flush()

            if training_episode % 1 == 0:#100
                model.save_weights(workspace_path + '/model/tree_supervised_model_' + str(training_episode))
                


    # Directory Setup
    # workspace_path= '/home/halim/minecraftRL/minecraft_bot_dev-master'
    # workspace_path= 'C:/Users/Halim/Downloads/minecraftRL/minecraft_bot_dev-master'
    # data_path='C:/Users/Halim/Downloads/minecraftRL/MineRLenv'

    # parser = argparse.ArgumentParser(description='Minecraft Supervised Learning')
    # parser.add_argument('--pretrained_model', type=str, help='pretrained model name')
    # parser.add_argument('--gpu_use', type=bool, default=False, help='use gpu')

    # arguments = parser.parse_args()

    # if arguments.gpu_use == True:
    #     gpus = tf.config.experimental.list_physical_devices('GPU')
    #     tf.config.experimental.set_virtual_device_configuration(gpus[0],
    #                 [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4000)])
    # else:
    #     os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


    # MineRL Setup
    env_name = 'MineRLTreechop'
    # writer = tf.summary.create_file_writer(workspace_path + "/tree_tensorboard")
    tree_data = minerl.data.make('MineRLTreechop-v0', data_dir=data_path)
    
    
    ##################
    class TreeTrajectoryDataset(tf.data.Dataset):
        def _generator(num_trajectorys):
            while True:
                trajectory_names = tree_data.get_trajectory_names()
                #print("len(trajectory_names): ", len(trajectory_names))
                
                #https://minerl.io/docs/api/data.html
                trajectory_name = random.choice(trajectory_names)
                print("trajectory_name: ", trajectory_name)
                
                trajectory = tree_data.load_data(trajectory_name, skip_interval=0, include_metadata=False)
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
    
        def __new__(cls, num_trajectorys=3):
            return tf.data.Dataset.from_generator(
                cls._generator,
                output_types=(tf.dtypes.float32, tf.dtypes.int32),
                args=(num_trajectorys,)
            )


    #################


    dataset = tf.data.Dataset.range(1).interleave(TreeTrajectoryDataset, 
    num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(1).prefetch(tf.data.experimental.AUTOTUNE)

    # num_actions = 43
    # num_hidden_units = 512

    #model = tf.keras.models.load_model('MineRL_SL_Model')
    model = network.ActorCritic(num_actions, num_hidden_units)

    if arguments.pretrained_model != None:
        print("Load Pretrained Model")
        model.load_weights("model/" + arguments.pretrained_model)

        
    # cce_loss = tf.keras.losses.CategoricalCrossentropy()
    # cce_loss_logits = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    # optimizer = tf.keras.optimizers.Adam(0.0001)

    for training_episode in range(0, 101): #2000000
        supervised_train(dataset, training_episode)


if __name__ == '__main__':
    main()