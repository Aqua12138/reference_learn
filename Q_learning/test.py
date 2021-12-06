#一维
import os
import numpy as np
import pandas as pd
import time
# MOTION_NUMBER = 6
# ACTION = ['right','left']
# MAX_EPISODE = 10
# EPSILON = 0.9
# GAMMA = 0.9
# ALPHA = 0.1
# def build_table():
#     # 创建全为o的空table
#     table = pd.DataFrame(np.zeros([MOTION_NUMBER,len(ACTION)]),columns=ACTION)
#     return table
# def env_update(S,episode, step_counter):
#     if S == 'terminal':
#         env = '第{}代,一共走了{}步'.format(episode+1,step_counter)
#     else:
#         env_list = ['-']*(MOTION_NUMBER-1)+['T']
#         env_list[S] = 'o'
#         env = ''.join(env_list)
#     print('\r{}'.format(env),end='')
#     time.sleep(0.3)
# def choose_action(S,action_table):
#     current_action = action_table.iloc[S,:]
#     if (np.random.uniform()>=EPSILON) or ((current_action==0).all()):
#         action = np.random.choice(ACTION)
#     else:
#         action = current_action.idxmax()
#     return action
# def feed_back_env(action,S):
#     if action == 'right':
#         if S == MOTION_NUMBER-2:
#             R = 1
#             S_ = 'terminal'
#         else:
#             R = 0
#             S_ = S+1
#     if action == 'left':
#         R = 0
#         if S == 0:
#             S_ = S
#         else:
#             S_ = S-1
#     return S_,R
# def main():
#     # 创建动作列表
#     action_table = build_table()
#     for episode in range(MAX_EPISODE):
#         S = 0
#         step_counter = 0
#         env_update(S,episode,step_counter)
#         is_terminal = False
#         while not is_terminal:
#             action = choose_action(S,action_table)
#             S_,R = feed_back_env(action,S)
#             q_predict = action_table.loc[S,action]
#             if S_ != 'terminal':

#                 q_target = R + GAMMA * action_table.iloc[S_, :].max()   # next state is not terminal
#             else:
#                 q_target = R     # next state is terminal
#                 is_terminal = True    # terminate this episode
#             action_table.loc[S, action] += ALPHA * (q_target - q_predict)  # update
#             S = S_  # move to next state
#             env_update(S, episode, step_counter + 1)
#             step_counter += 1
#     return action_table

#二维
Weight = 5
Height = 5
Action = ['right','left','down','up']
Max_episode = 1
EPSILON = 0.9
GAMMA = 0.9
ALPHA = 0.2
Sleep_time = 0.1

def env_update(episode,step,observation):
    if observation == 'terminal':
        print('\rThe {} episode: {} steps'.format(episode+1,step),end='')
        time.sleep(2)
    else:
        interface = [['*', '*', '*', '*', '*', '*'],
                     ['*', '*', '*', '*', '*', '*'],
                     ['*', '*', '*', '*', '*', '*'],
                     ['*', '*', '*', '*', '*', '*'],
                     ['*', '*', '*', '*', '*', '*'],
                     ['*', '*', '*', '*', '*', 'T']]
        interface[observation[0]][observation[1]] = 'o'
        for i in interface:
            print(''.join(i))
        time.sleep(Sleep_time)
        os.system('clear')
def choose_action(state,q_table):
    q_table = check_if_state_exit(state,q_table)
    current_action = q_table.loc[str(state),:]
    if (np.random.uniform()>EPSILON) or ((current_action==0).all()):
        action = np.random.choice(Action)
    else:
        action = np.random.choice(current_action[current_action == np.max(current_action)].index)#!
    return action,q_table
def check_if_state_exit(state,q_table):
    if str(state) not in q_table.index:
        q_table = q_table.append(pd.Series(np.zeros([len(Action)]),index=q_table.columns,name=str(state)))
    return q_table
def feed_back_action(S,action,q_table):
    if action == 'right':
        if S == [Height,Weight-1]:
            S_ = 'terminal'
            R = 1
        else:
            R = 0
            if S[1] != Weight:
                S_ = [S[0], S[1]+1]
            else:
                S_ = S
    elif action == 'left':
        R = 0
        if S[1]==0:
            S_ = S
        else:
            S_ = [S[0],S[1]-1]
    elif action == 'down':
        if S == [Height-1,Weight]:
            S_ = 'terminal'
            R = 1
        else:
            R = 0
            if S[0] != Height:
                S_ = [S[0]+1,S[1]]
            else:
                S_ = S
    elif action == 'up':
        R = 0
        if S[0]==0:
            S_ = S
        else:
            S_ = [S[0]-1,S[1]]
    table = check_if_state_exit(S_,q_table)
    return table,S_ ,R
def main():
    #创建Table
    if os.path.exists('/Users/zhx/Documents/test_table.csv'):
        q_table = pd.read_csv('/Users/zhx/Documents/test_table.csv',index_col=0)
    else:
        q_table = pd.DataFrame(np.zeros([1, len(Action)]), columns=Action)
    for episode in range(Max_episode):
        step = 0#步数
        observation = [0,0]#位置
        env_update(episode,step,observation)#更新初始化
        is_terminal = False#终点信号
        while not is_terminal:
            action,q_table = choose_action(observation,q_table)
            q_table,observation_ ,R = feed_back_action(observation,action,q_table)
            q_predict = q_table.loc[str(observation),action]
            if observation_ != 'terminal':
                q_target = R + GAMMA * q_table.loc[str(observation_),:].max()
            else:
                q_target = R
                is_terminal = True
            q_table.loc[str(observation),action] = ALPHA*(q_target-q_predict)
            observation = observation_
            step += 1
            env_update(episode,step,observation)
    return q_table




if __name__ == '__main__':
    table = main()
    print(table)
    table.to_csv('/Users/zhx/Documents/test_table.csv')
