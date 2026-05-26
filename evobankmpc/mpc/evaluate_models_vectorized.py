import numpy as np


def evaluate_models_vectorized(models, n_models, current_state, input_val, Ts, params):



    n_models = len(models)


    Bfs, Cfs, Dfs, Brs, Crs, Drs = params


    x0_batch = np.tile(current_state, (n_models, 1))  # shape: (n_models, state_dim)
    

    u_batch = np.tile(input_val, (n_models, 1))      # shape: (n_models, input_dim)


    batch_model = models[0].__class__(
        Bf=Bfs, Cf=Cfs, Df=Dfs,    
        Br=Brs, Cr=Crs, Dr=Drs,    
        mass=models[0].mass,       
        lf=models[0].lf,          
        lr=models[0].lr,           
        Iz=models[0].Iz,         
        Cm1=models[0].Cm1,         
        Cm2=models[0].Cm2,        
        Cr0=models[0].Cr0,        
        Cr2=models[0].Cr2,         
        input_acc=models[0].input_acc,  
        approx=models[0].approx        
    )


    return np.vstack(batch_model._integrate_batch(x0_batch, u_batch, 0, Ts))[:, 0:4]