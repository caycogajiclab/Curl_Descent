import numpy as np
#import matplotlib.pyplot as plt
#from matplotlib import colors as clr
#import matplotlib.animation as animation
from tqdm import tqdm
#from scipy.optimize import minimize
#from scipy import optimize
#from jax.scipy.optimize import minimize
#from jax.scipy import optimize
#from scipy.linalg import circulant
#from scipy import fft
from joblib import Parallel, delayed
#import torch
#from cycler import cycler
#import torchvision
#import cProfile
#from scipy import special
#from scipy.linalg import qr
#from scipy import sparse
import sys
import pickle 

#import plotly.graph_objects as go
#from plotly.subplots import make_subplots


if len(sys.argv) < 2:
    print("Usage: python tanh_ts_param.py <argument>")
    print(len(sys.argv))
    sys.exit(1)

# Get the argument from the command-line
task_id = int(sys.argv[1])

# Read the argList.txt file
arg_list_file = "argList.txt"
with open(arg_list_file, 'r') as file:
    lines = file.readlines()

# Ensure the task_id is within the range of available lines
if task_id < 0 or task_id >= len(lines):
    print(f"Error: task_id {task_id} is out of range for the argList file.")
    sys.exit(1)

# Get the argument corresponding to the task ID
arguments = lines[task_id+1].strip().split()


nb_data_points, nb_test_data_points, input_dim, teacher_size, student_size, output_dim =  map(int, arguments[0:6])
init_range = float(arguments[6])
batch_size, nb_AH_layer2, nb_AH_layer1 = map(int,arguments[7:10])
nb_epochs, learning_rate, feedback_scale = map(float, arguments[10:13])
nb_init = int(arguments[13])
nonlinearity, modification_type, initialization_type = map(str, arguments[14:17])

nb_epochs = int(nb_epochs)
######
def log_downsample_indices(total_points, num_points=1000):
    """
    Generate logarithmically spaced indices for downsampling.
    The indices are computed in the range [1, total_points] then converted to zero-indexed values.
    """
    indices = np.unique(np.round(np.geomspace(1, total_points, num=num_points)).astype(int)) - 1
    return indices

class TeacherStudentLearning:
    def __init__(self, input_dim, teacher_size, student_size, teacher_params, student_params_ini):
        self.input_dim = input_dim
        self.teacher_size = teacher_size
        self.student_size = student_size
        
        self.teacher_params = teacher_params.copy()
        self.student_params = student_params_ini.copy()
        
        # Initialize batch normalization parameters
        self.student_params['gamma1'] = np.ones(self.student_size)/2
        self.student_params['beta1'] = np.zeros(self.student_size)
        self.student_params['gamma2'] = np.ones(1)/2
        self.student_params['beta2'] = np.zeros(1)
        
        # Initialize batch normalization parameters
        self.teacher_params['gamma1'] = np.ones(self.teacher_size)/2
        self.teacher_params['beta1'] = np.zeros(self.teacher_size)
        self.teacher_params['gamma2'] = np.ones(1)/2
        self.teacher_params['beta2'] = np.zeros(1)



    def phi(self, x, nonlinearity):
        if nonlinearity == 'lin':
            return x
        elif nonlinearity == 'tanh':
            return np.tanh(x)
        elif nonlinearity == 'relu':
            return np.clip(x, a_min=0, a_max=None)
        elif nonlinearity == 'sigmoid':
            return 1/(1+np.exp(-x))
        else:
            raise ValueError(f"Unknown nonlinearity: {nonlinearity}")

    def phi_p(self, x, nonlinearity):
        if nonlinearity == 'lin':
            return np.full_like(x, 1)
        elif nonlinearity == 'tanh':
            return 1 - np.tanh(x) ** 2
        elif nonlinearity == 'relu':
            return (x > 0) * np.full_like(x, 1)
        elif nonlinearity == 'sigmoid':
            return np.exp(x)/(1+np.exp(x))**2
        else:
            raise ValueError(f"Unknown nonlinearity: {nonlinearity}")

    def ff_pass(self, x, network_params, nonlinearity, batch_norm=True):
        W1, W2 = network_params['W1'], network_params['W2']
        
        if batch_norm:
            a1 = np.dot(W1, x)
            a1_bn = ((a1-a1.mean(axis = -1)[:,None])/(a1.std(axis = -1)[:,None]))*network_params['gamma1'][:,None]+network_params['beta1'][:,None]
            h = self.phi(a1_bn, nonlinearity)
            
            a2 = np.dot(W2, h)
            a2_bn = ((a2-a2.mean(axis = -1)[:,None])/(a2.std(axis = -1)[:,None]))*network_params['gamma2'][:,None]+network_params['beta2'][:,None]
            y = self.phi(a2_bn, nonlinearity)
        else:
            h = self.phi(np.dot(W1, x), nonlinearity)
            y = self.phi(np.dot(W2, h), nonlinearity)
        return y
            

    
    def error(self, input_list, nonlinearity, batch_norm):
        inputs = np.stack(input_list, axis=1)  # Combine inputs for batch processing
        y_hat = self.ff_pass(inputs, self.student_params, nonlinearity, batch_norm)
        y = self.ff_pass(inputs, self.teacher_params, nonlinearity, batch_norm)
        return 0.5 * np.mean((y_hat - y) ** 2)
    
    def learning_update(self, x, B, learning_rate, plasticity, modification_type, nonlinearity, batch_norm, ):
        y_hat = self.ff_pass(x, self.student_params, nonlinearity, batch_norm)
        y = self.ff_pass(x, self.teacher_params, nonlinearity, batch_norm)
        e = y_hat - y
        
        if batch_norm :
            a = np.dot(self.student_params['W1'],  x)
            a = ((a-a.mean(axis = -1)[:,None])/(a.std(axis = -1)[:,None]))*self.student_params['gamma1'][:,None]+self.student_params['beta1'][:,None]
            h = self.phi(a, nonlinearity)   
        else:
            a = self.student_params['W1'] @ x
            h = self.phi(a, nonlinearity)
        
        
        if plasticity == 'GD':
            da2 = e
            da1 = (self.student_params['W2'].T @ da2) * self.phi_p(a, nonlinearity)
        elif plasticity == 'Hebb_DFA_inh_w_GD_feedback':
            da2 = e
            da1 = (self.student_params['W2'].T @ da2) * self.phi_p(a, nonlinearity)
        elif plasticity == 'DFA':
            da2 = e
            da1 = (B @ da2) * self.phi_p(a, nonlinearity)
        else:
            raise ValueError(f"Unknown plasticity: {plasticity}")

        if modification_type == 'col':
        # Compute weight updates by summing over the batch dimension (axis=1)
            dWs2 = -learning_rate * (da2 @ h.T @ self.student_params['D2']) / da2.shape[1]
            dWs1 = -learning_rate * (da1 @ x.T @ self.student_params['D1']) / da1.shape[1]
        
        elif modification_type == 'row':
        # Update on row instead
            dWs2 = -learning_rate * (self.student_params['D2'] @ da2 @ h.T) / da2.shape[1]
            dWs1 = -learning_rate * ( self.student_params['D1'] @ da1 @ x.T) / da1.shape[1]
        
        return dWs1, dWs2
    
    def _normalize_weights(self, weights):
        # Normalize each row of the weight matrix to have a unit norm
        norms = np.linalg.norm(weights, axis=1, keepdims=True)  # L2 norm of each row
        return weights / (norms + 1e-8)  # Add epsilon to prevent division by zero



    def learning_simu_batch(self, input_list, input_test_list, batch_size, nb_epochs, B, learning_rate, plasticity, modification_type, nonlinearity, normalize_weights=True, batch_norm=True,  test_set=False):
        """
        Simulates the learning process with batch updates.

        Args:
            input_list: List of input samples.
            batch_size: Size of each batch.
            nb_epochs: Number of training epochs.
            B: Feedback alignment matrix.
            learning_rate: Learning rate.
            plasticity: Type of plasticity ('GD' or 'DFA').
            nonlinearity: Nonlinearity function.
            normalize_weights: Whether to normalize weights after each batch update.

        Returns:
            Ws1_hist: History of W1 updates.
            Ws2_hist: History of W2 updates.
            error_hist: History of error.
        """
        inputs = np.stack(input_list, axis=1)  # Stack inputs into a single matrix (shape: input_dim x total_samples)
        total_samples = inputs.shape[1]

        Ws1_hist = [self.student_params['W1'].copy()]
        Ws2_hist = [self.student_params['W2'].copy()]
        
        norm_w1_ini = np.linalg.norm(self.student_params['W1'].copy())
        norm_w2_ini = np.linalg.norm(self.student_params['W2'].copy())
        
        error_hist = [self.error(input_list[:batch_size], nonlinearity, batch_norm)]
        if test_set:
            test_error_hist = [self.error(input_test_list, nonlinearity, batch_norm)]
        else:
            test_error_hist = np.zeros_like(np.array(error_hist))

            
            
        log_indices = log_downsample_indices(nb_epochs, num_points=100)
        ind = 1
        
        for epoch in range(nb_epochs):
            # Shuffle the data at the start of each epoch
            indices = np.random.permutation(total_samples)
            shuffled_inputs = inputs[:, indices]

            for start_idx in range(0, total_samples, batch_size):
                # Define batch range
                end_idx = min(start_idx + batch_size, total_samples)
                batch = shuffled_inputs[:, start_idx:end_idx]
                
                dWs1_avg, dWs2_avg = self.learning_update(batch, B, learning_rate, plasticity, modification_type, nonlinearity, batch_norm)

                # Update weights
                self.student_params['W1'] += dWs1_avg
                self.student_params['W2'] += dWs2_avg
                
                norm_w1 = np.linalg.norm(self.student_params['W1'])
                norm_w2 = np.linalg.norm(self.student_params['W2'])
                
                self.student_params['W1'] *= norm_w1_ini / norm_w1
                self.student_params['W2'] *= norm_w2_ini / norm_w2
                
                # Enforce weight normalization
                if normalize_weights:
                    self.student_params['W1'] = self._normalize_weights(self.student_params['W1'])
                    self.student_params['W2'] = self._normalize_weights(self.student_params['W2'])

            if epoch == log_indices[ind]:
                # Record history
                Ws1_hist.append(self.student_params['W1'].copy())
                Ws2_hist.append(self.student_params['W2'].copy())

                error_hist.append(self.error(input_list, nonlinearity, batch_norm))  # Evaluate on the first sample of the batch [batch[:, 0]]
                if test_set:
                    test_error_hist.append(self.error(input_test_list, nonlinearity, batch_norm))

                ind+=1


        return np.array(Ws1_hist), np.array(Ws2_hist), np.array(error_hist), np.array(test_error_hist)


    
    def learning_simu(self, input_list, B, learning_rate, plasticity, nonlinearity):
        Ws1_hist = [self.student_params['W1'].copy()]
        Ws2_hist = [self.student_params['W2'].copy()]
        error_hist = [self.error([input_list[0]], nonlinearity)]
        
        for x in input_list:
            dWs1, dWs2 = self.learning_update(x, B, learning_rate, plasticity, nonlinearity)
            self.student_params['W1'] += dWs1
            self.student_params['W2'] += dWs2
            
            Ws1_hist.append(self.student_params['W1'].copy())
            Ws2_hist.append(self.student_params['W2'].copy())
            error_hist.append(self.error([x], nonlinearity))
        
        return np.array(Ws1_hist), np.array(Ws2_hist), np.array(error_hist)

    
    
    
#######
# PREPARING THE SIMU RUN #
######

label_list = ['GD', 'DFA', 'DFA with inhibitory', 'Hebb DFA inh with GD feedback']
plasticity_list = ['GD', 'DFA', 'DFA', 'Hebb_DFA_inh_w_GD_feedback']
nonlinearity_list = [nonlinearity, nonlinearity, nonlinearity, nonlinearity]

Ws1_hist_dic = {}
Ws2_hist_dic = {}
error_hist_dic = {}
for i,label in enumerate(label_list):
    Ws1_hist_dic[label] = []
    Ws2_hist_dic[label] = []
    error_hist_dic[label] = []

if modification_type == 'col':
    # COL Recreate identity matrices (if needed) and modify Ds2, Dt2  COL
    Ds1 = np.eye(input_dim)
    Ds2 = np.eye(student_size)
    Dt1 = np.eye(input_dim)
    Dt2 = np.eye(teacher_size)
elif modification_type == 'row':
    # ROW Recreate identity matrices (if needed) and modify Ds2, Dt2  ROW
    Ds1 = np.eye(student_size)
    Ds2 = np.eye(output_dim)
    Dt1 = np.eye(teacher_size)
    Dt2 = np.eye(output_dim)

for i in range(nb_AH_layer1):
    Ds1[i,i] = -1
    Dt1[i,i] = -1
for i in range(nb_AH_layer2):
    Ds2[i,i] = -1
    Dt2[i,i] = -1


teacher_params_mat = []
for j in range(nb_init):
    
    
    if initialization_type == 'normal':
        #Wt1 = np.random.normal(0, 1, size=(teacher_size, input_dim))
        #Wt2 = np.random.normal(0, 1, size=(1, teacher_size))
        Wt1 = np.random.normal(0, 1/(np.sqrt(input_dim)), size=(teacher_size, input_dim))
        Wt2 = np.random.normal(0, 1/(np.sqrt(teacher_size)), size=(output_dim, teacher_size))
    elif initialization_type == 'uniform':
        #Wt1 = np.sqrt(1) * np.random.uniform(-1, 1, size=(teacher_size, input_dim))
        #Wt2 = np.sqrt(1) * np.random.uniform(-1, 1, size=(output_dim, teacher_size))
        Wt1 = np.sqrt(1) * np.random.uniform(-1/(np.sqrt(input_dim)), 1/(np.sqrt(input_dim)), size=(teacher_size, input_dim))
        Wt2 = np.sqrt(1) * np.random.uniform(-1/(np.sqrt(teacher_size)), 1/(np.sqrt(teacher_size)), size=(output_dim, teacher_size))
    
    # Define the list of teacher parameters for each label index
    teacher_params_list = [
        {'W1': Wt1, 'W2': Wt2, 'D1': np.eye(input_dim), 'D2': np.eye(teacher_size)},
        {'W1': Wt1, 'W2': Wt2, 'D1': np.eye(input_dim), 'D2': np.eye(teacher_size)},
        {'W1': Wt1, 'W2': Wt2, 'D1': np.eye(input_dim), 'D2': np.eye(teacher_size)},
        {'W1': Wt1, 'W2': Wt2, 'D1': np.eye(input_dim), 'D2': np.eye(teacher_size)}
    ]
    teacher_params_mat.append(teacher_params_list)


# Initialize student weights for all trials
student_params_ini_mat = []
for j in range(nb_init):
    
    if initialization_type == 'normal':
        Ws1_ini = np.random.normal(0, init_range/(np.sqrt(input_dim)), size=(student_size, input_dim))
        Ws2_ini = np.random.normal(0, init_range/(np.sqrt(student_size)), size=(output_dim, student_size))
    if initialization_type == 'uniform':
        Ws1_ini = np.sqrt(1) * np.random.uniform(-init_range/(np.sqrt(input_dim)), init_range/(np.sqrt(input_dim)), size=(student_size, input_dim))
        Ws2_ini = np.sqrt(1) * np.random.uniform(-init_range/(np.sqrt(student_size)), init_range/(np.sqrt(student_size)), size=(output_dim, student_size))
        
    if modification_type == 'col':
        # Similarly, define the list of initial student parameters
        student_params_ini_mat.append([
            {'W1': Ws1_ini.copy(), 'W2': Ws2_ini.copy(), 'D1': np.eye(input_dim), 'D2': np.eye(student_size)},
            {'W1': Ws1_ini.copy(), 'W2': Ws2_ini.copy(), 'D1': np.eye(input_dim), 'D2': np.eye(student_size)},
            {'W1': Ws1_ini.copy(), 'W2': Ws2_ini.copy(), 'D1': Ds1, 'D2': Ds2},
            {'W1': Ws1_ini.copy(), 'W2': Ws2_ini.copy(), 'D1': Ds1, 'D2': Ds2}
        ])
    elif modification_type == 'row':
        student_params_ini_mat.append([
            {'W1': Ws1_ini.copy(), 'W2': Ws2_ini.copy(), 'D1': np.eye(student_size), 'D2': np.eye(output_dim)},
            {'W1': Ws1_ini.copy(), 'W2': Ws2_ini.copy(), 'D1': np.eye(student_size), 'D2': np.eye(output_dim)},
            {'W1': Ws1_ini.copy(), 'W2': Ws2_ini.copy(), 'D1': Ds1, 'D2': Ds2},
            {'W1': Ws1_ini.copy(), 'W2': Ws2_ini.copy(), 'D1': Ds1, 'D2': Ds2}
        ])
    
    
    
    
    
# Create input list
input_list = [1/np.sqrt(2)*np.random.normal(0, 1, size=input_dim) for _ in range(nb_data_points)] #1/(np.sqrt(input_dim)

input_test_list = [1/np.sqrt(2)*np.random.normal(0, 1, size=input_dim) for _ in range(nb_test_data_points)] #1/(np.sqrt(input_dim)

    
def simulation_run(i, j, input_list, input_test_list, student_params_ini_mat, teacher_params_mat):


    student_params_ini_list = student_params_ini_mat[j]
    teacher_params_list = teacher_params_mat[j]

    # Select parameters corresponding to label index i
    teacher_params = teacher_params_list[i]
    student_params_ini = student_params_ini_list[i]
    
    
    # Create and scale B matrix
    #B = np.random.normal(0,1/(np.sqrt(teacher_size)*0.01), size = (student_size, 1)) # np.array([[1,1,1,1]]).T   #
    #B = 0.1 * torch.rand(student_size, 1) * 2 / np.sqrt(student_size) - 1 / np.sqrt(student_size)
    B = feedback_scale *  np.sqrt(1) * np.random.uniform(- 1/(np.sqrt(student_size)), 1/(np.sqrt(student_size)), size =(student_size, output_dim))
    #B = np.random.normal(0, 1, size=(student_size, output_dim))
    #B = 1*np.abs(B) * np.linalg.norm(student_params_ini['W2']) / np.linalg.norm(B)
    #B = np.abs(B)
    #B *= np.sign(student_params_ini['W2'].T)
    
    
    # Create the learning system and run the simulation
    learning_system = TeacherStudentLearning(input_dim, teacher_size, student_size,
                                               teacher_params, student_params_ini)
    
    Ws1_hist, Ws2_hist, error_hist, test_error_hist = learning_system.learning_simu_batch(
        input_list,
        input_test_list,
        batch_size=batch_size, #len(input_list),  # or nb_samples if defined differently
        nb_epochs=nb_epochs,
        B=B,
        learning_rate=learning_rate*batch_size/len(input_list),
        plasticity=plasticity_list[i],
        modification_type = modification_type,
        nonlinearity=nonlinearity_list[i],
        normalize_weights=False,
        batch_norm=False,
        test_set = True
    )
    
    # Return the label index and the histories from this trial
    return i, Ws1_hist, Ws2_hist, error_hist, test_error_hist, B

# Parallelize over both loops. Each task corresponds to one simulation_run call.
results = Parallel(n_jobs=-1)(
    delayed(simulation_run)(i, j, input_list, input_test_list, student_params_ini_mat, teacher_params_mat)
    for i in range(len(label_list))
    for j in tqdm(range(nb_init)))

# Initialize dictionaries to aggregate the simulation histories
Ws1_hist_dic = {label: [] for label in label_list}
Ws2_hist_dic = {label: [] for label in label_list}
error_hist_dic = {label: [] for label in label_list}
test_error_hist_dic = {label: [] for label in label_list}
B_dic = {label: [] for label in label_list}


# Aggregate results from parallel execution
for i, Ws1_hist, Ws2_hist, error_hist, test_error_hist, B in results:
    Ws1_hist_dic[label_list[i]].append(Ws1_hist)
    Ws2_hist_dic[label_list[i]].append(Ws2_hist)
    error_hist_dic[label_list[i]].append(error_hist)
    test_error_hist_dic[label_list[i]].append(test_error_hist)
    B_dic[label_list[i]].append(B)
    
params_dic = {'nb_data_points':nb_data_points, 'nb_test_data_points':nb_test_data_points,
             'input_dim':input_dim, 'teacher_size':teacher_size, 'student_size':student_size, 'output_dim':output_dim,
             'nb_AH_layer1':nb_AH_layer1, 'nb_AH_layer2':nb_AH_layer2, 'nb_epochs':nb_epochs,
             'nb_init':nb_init, 'nonlinearity_list':nonlinearity_list, 'learning_rate':learning_rate ,
             'B_dic':B_dic, 'batch_size':batch_size, 'init_range':init_range, 'modification_type':modification_type,
             'initialization_type':initialization_type, 'feedback_scale':feedback_scale, 'batch_size':batch_size}

#np.save(f'/home/hninou/linear_ts/run_mult_{task_id}.npy', (params_dic, input_list, input_test_list, Ws1_hist_dic, Ws2_hist_dic, error_hist_dic, test_error_hist_dic), allow_pickle = True)


# Save using pickle
with open(f'/shared/projects/project_curlDynamics/clean_code/normalized/phase_diag_L1_tanh/run_{nb_AH_layer1}-{input_dim}__{nb_AH_layer2}-{student_size}__{task_id}.pkl', 'wb') as f:
    pickle.dump((params_dic, np.array(input_list), np.array(input_test_list), error_hist_dic, test_error_hist_dic, teacher_params_mat), f)