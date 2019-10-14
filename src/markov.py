def get_transition_matrix(df_traj):
        
    transition_matrix = np.zeros((1024,1024))

    
    for index, row in df_traj.drop(df_traj.tail(1).index).iterrows():

        x = df_traj.loc[index, 'reg_id'] - 1
        y = df_traj.loc[index+1, 'reg_id'] - 1

        transition_matrix[x,y] =+ 1


    transition_matrix = transition_matrix/(df_traj.shape[0]-1)
    
    return transition_matrix

def markov(df_anon, df_ref):
    transition_matrix = get_transition_matrix(df_ref)
    
    new_reg_id = list()
    
    df_result = df_anon.copy()
    
    for user_id in df_anon.pse_id.unique():
        df_user = df_anon[df_anon.pse_id == user_id].reset_index(drop=True)
        new_reg_id.append(df_user.loc[0, 'reg_id'])

        for index, row in df_user.drop(df_user.tail(1).index).iterrows():
            x = df_user.loc[index, 'reg_id'] - 1
            y = df_user.loc[index+1, 'reg_id'] - 1

            if (transition_matrix[x,y] != 0):
                new_reg_id.append(y+1)
            else:
                new_reg_id.append(x+1)
    
    df_result['markov_reg_id'] = new_reg_id
    
    return df_result