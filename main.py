'''---------------USER PARAMETERS---------------'''

''' SIMULATION PARAMETERS '''
# Dimensions of the square lattice
N = 100

# Probability that a dipole starts off in the +1 state
thresh = 0.75

# Exchange energy constant (Arbitrary Units)
J = 1

# Thermodynamic Beta (1/J Units)
B = 1

# Number of simulation steps to take
n = 100000

'''VISUALIZATION PARAMETERS'''

# Render FPS
fps = 60

# Number of frames to render
nframes = 200

# Save all simulated frames?
save_frames = False

'''--------------------------------------------'''

''' IMPORTS '''
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation  

''' FUNCTIONS '''

# Define lattice
lattice = np.array([[1 if j<thresh else -1 for j in i] for i in np.random.rand(N,N)])

# Calculate energy of the lattice (E/J)
def energy(lattice, kernel = np.array([[0, 1, 0],
									   [1, 0, 1],
									   [0, 1, 0]])):
	return -1 * np.sum(lattice * ndimage.convolve(lattice, kernel, mode='constant',cval=0))

# Propose and select a new state by flipping one dipole
def newState(prev_frame):
	# Get previous state
	state = prev_frame[0]

	# Create new state
	i, j = np.random.randint(N,size=2)
	update = np.ones_like(state)
	update[i][j] = -1
	state_new = update * state

	# Calculate energies
	e_old = prev_frame[1]
	e_new = energy(state_new)

	# Calculate change in energy
	E_diff = e_new - e_old

	# If proposed state is lower in energy, keep the move
	if E_diff <= 0:
		return state_new, e_new
	# If proposed state is higher in energy
	elif np.random.rand() < np.exp(-1 * B * E_diff):
		return state_new, e_new
	else:
		return state, e_old

def simulate(i, save_frames = True):
	'''
	i: Number of simulation frames
	save_frames: Save all lattice frames
	'''

	lattice_frames = [(lattice,energy(lattice))]

	# Save all lattice frames
	if save_frames:
		for _ in range(i):
			print(f"Step {_}/{i}, {int(100*_/i)}% Complete...")
			lattice_new, e_new = newState(lattice_frames[_])

			# Add a new updated frame
			lattice_frames.append((lattice_new, e_new))

	else:
		for _ in range(i):
			print(f"Step {_}/{i}, {int(100*_/i)}% Complete...")
			lattice_new, e_new = newState(lattice_frames[0])

			# Add a new updated frame
			lattice_frames.append((lattice_new, e_new))

			# Delete the now old-old frame
			lattice_frames.pop(0)

	return lattice_frames

def animate(i, lattice_frames, lat, nframes):
	f = int(len(lattice_frames) * i/nframes)
	print("Frame: ", i, "| Energy: ", lattice_frames[f][1], "| Net Spin: ", np.sum(lattice_frames[f][0]))
	lat.set_data(lattice_frames[f][0])

def animate_ising(lattice_frames, nframes = 1000, fps=60):
	'''
	Animates the evolution of the lattice towards equilibrium

	lattice_frames: Python list of tuples of form (lattice_frame, energy)
	nframes: Number of frames to render in the animation (nframes < n)
	fps: Render FPS
	'''

	fig = plt.figure()  
	lat = plt.imshow(lattice)
	anim = FuncAnimation(fig, animate,
	                     frames = nframes, interval = int(1000/fps), fargs = [lattice_frames, lat, nframes])
	plt.show()	

if __name__ == "__main__":
	lattice_frames = simulate(n, save_frames)
	print(lattice_frames)
	#animate_ising(lattice_frames, nframes)
