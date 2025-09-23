import numpy as np

def generate_2D_double_vortex_flow_field(N_POINTS_X=32, N_POINTS_Y=16, strength1=0.20, strength2=.10):
    # Create a grid
    x = np.linspace(0, 1, N_POINTS_X)
    y = np.linspace(0, 1, N_POINTS_Y)
    X, Y = np.meshgrid(x, y)

    # Initialize velocity components
    u = np.zeros((N_POINTS_Y, N_POINTS_X))
    v = np.zeros((N_POINTS_Y, N_POINTS_X))

    # Define the centers of the vortices
    center1 = (0.25, 0.5)
    center2 = (0.75, 0.5) 

    # Calculate the flow field for the first vortex
    r1_x = X - center1[0]
    r1_y = Y - center1[1]
    r1 = np.sqrt(r1_x**2 + r1_y**2)
    
    u[:, :N_POINTS_X // 2] = -strength1 * r1_y[:, :N_POINTS_X // 2] / (r1[:, :N_POINTS_X // 2] + 1e-8)
    v[:, :N_POINTS_X // 2] = strength1 * r1_x[:, :N_POINTS_X // 2] / (r1[:, :N_POINTS_X // 2] + 1e-8)

    # Calculate the flow field for the second vortex
    r2_x = X - center2[0]
    r2_y = Y - center2[1]
    r2 = np.sqrt(r2_x**2 + r2_y**2)
    
    u[:, N_POINTS_X // 2:] = strength2 * r2_y[:, N_POINTS_X // 2:] / (r2[:, N_POINTS_X // 2:] + 1e-8)
    v[:, N_POINTS_X // 2:] = -strength2 * r2_x[:, N_POINTS_X // 2:] / (r2[:, N_POINTS_X // 2:] + 1e-8)

    advection_velocity = np.stack((u,v), axis=-1)

    return advection_velocity

def generate_1D_initial_density(N_POINTS_X=32,ambient_density=0.1):
    density = np.ones(N_POINTS_X)*ambient_density
    density[int(N_POINTS_X/2)-3:int(N_POINTS_X/2)+3] = 0.2

    return density

def generate_2D_initial_density(N_POINTS_X=32,N_POINTS_Y=16,ambient_density=0.1):

    density = np.ones([N_POINTS_Y,N_POINTS_X])*ambient_density
    density[int(N_POINTS_Y/2)-3:int(N_POINTS_Y/2)+3,int(N_POINTS_X/2)-3:int(N_POINTS_X/2)+3] = 0.2

    return density



