# -*- coding: utf-8 -*-
"""
Spyder Editor

MENG 443 plotter

Use Apple screen capture to record animation in console
https://support.apple.com/en-us/HT208721 Shift, Cmd, 5
"""

import numpy as np
from matplotlib import pyplot as plt

# State of pogo: [x, x_dot, y, y_dot, theta, theta_dot]
# upwards is positive

class Pogo_robot:
    def __init__(self, k, M, l0, init_x, init_xdot, init_y, init_ydot, init_theta, init_thetadot):
        """Initialize state parameters and constant"""
        self.x = init_x
        self.xdot = init_xdot
        self.y = init_y
        self.ydot = init_ydot
        self.theta = init_theta   # radian
        self.thetadot = init_thetadot
        self.k = k
        self.M = M
        self.lk = l0   # maybe check lengths, instantaneous
        self.l0 = l0   # default length
        self.ldot = 0  # initialize with no compression 
        print("Pogo initiated. Spring constant", k, ", mass", M,\
              ", leg length", l0)
        print("Initial [x, xdot, y, ydot, theta, thetadot]:", self.x, self.xdot,\
             self.y, self.ydot, self.theta, self.thetadot)
        
        # Check not initially in contact
        foot_y = self.y - self.lk*np.cos(self.theta - np.pi/2)
        if foot_y < 0:
            print("Pogo seems to be in contact already with ground, do you want to reset it?")
        else:
            print("Ready to hop!")
            print("\|/-~~~~~~~~~~")
    
    def flight(self):
        """Free flight phase (no ground contact)"""
        prev_state = np.array([self.x, self.xdot, self.y, self.ydot, self.theta, self.thetadot, self.lk, self.ldot])    
        self.x += prev_state[1]*dT   # Euler timestep integrate
        self.xdot += 0.0
        self.y += prev_state[3]*dT
        self.ydot += -9.81*dT
        self.theta += prev_state[5]*dT
        self.thetadot += 0  # no change
        self.lk = self.l0   # back to equilibrium point
        self.ldot = 0   
        return 
  
    def is_in_contact(self):
        """
        Flag function to compute if foot lands on the 
        ground (not takeoff check)
        """ 
        foot_y = self.y - self.lk*np.cos(self.theta - np.pi/2)
        #print("Debug is_in_contact: foot_y", np.round(foot_y,3))
        if foot_y <= 0:
            print("    Incident x dot:", self.xdot)
            print("    Incident y dot:", np.round(self.ydot,3))
            return True
        else:
            return False 

    def first_contact(self):
        """
        Function with conversion of linear to rotational velocities,
        run once only per contact
        """
        #print("Debugging: contact")
        total_v = ((self.ydot)**2+(self.xdot)**2)**0.5
            
        # tangential (angular) velocity at impact
        v_tangent = -np.abs(self.ydot)*np.sin(self.theta-np.pi/2) + \
                   self.xdot*np.sin(np.pi - self.theta)
        self.thetadot = v_tangent/self.l0
        #print("    Debugging contact: theta_dot", np.round(self.thetadot,3))
        
        # radial velocity at impact
        self.ldot = -abs(self.ydot)*np.cos(self.theta-np.pi/2) - \
                    self.xdot*np.cos(np.pi - self.theta)
        #print("    Debugging contact: l_dot", np.round(self.ldot,3))
        polar_coord_speed = (v_tangent**2 + self.ldot**2)**0.5
        
        if abs((polar_coord_speed/total_v)-1) > 0.01:
            print("Impact error: velocity off")
            return
        # This check is good
        #print("  Check: impact speed", total_v, "post-impact speed", polar_coord_speed)
        
        return
    
    def contact(self):
        """Contacted mechanics with the ground"""
        prev_state = np.array([self.x, self.xdot, self.y, self.ydot, self.theta, self.thetadot, self.lk, self.ldot])    
    
        # l_ddot formula: -k/M(l_k - l_k0) - g sin(theta) + l*theta_dot^2 
        l_ddot = -self.k/self.M*(prev_state[6]-self.l0) - 9.81*np.sin(prev_state[4]) + prev_state[6]*(prev_state[5])**2
        print("    l_ddot:",np.round(l_ddot,3))
        
        # theta_ddot formula: -2*l_dot/l*theta_dot - g/l cos(theta)
        theta_ddot = -2*prev_state[7]/prev_state[6]*prev_state[5] - 9.81/prev_state[6]*np.cos(prev_state[4])
        print("    theta_ddot:",np.round(theta_ddot,3))
    
        # calculate l, theta 
        self.theta    += prev_state[5]*dT  # use previous step's accelerations
        self.thetadot += theta_ddot*dT
        self.lk       += prev_state[7]*dT
        self.ldot     += l_ddot*dT   
        
        print("    theta:", np.round(self.theta,3))
        print("    lk:", np.round(self.lk,3))
    
        # Update x, y and linear velocities, pivoting
        # TODO: "tangential" velocity wrong --> need to add "radial" component too
        
        v_tan = abs(prev_state[6]*prev_state[5])  # tangential velocity 
        self.x += prev_state[1]*dT   # Euler timestep integrate
        self.xdot = v_tan*np.cos(prev_state[4]-np.pi/2)
        self.y += prev_state[3]*dT
        self.ydot = v_tan*np.sin(prev_state[4]-np.pi/2)
    
        print("    x:", self.x)
        print("    y:", self.y)
        return
    
   
    def takeoff_check(self):
        """
        Pogo ground release check function
        Return true if spring is back to equilibrium extension 
        (to back to flight phase)
        """
        print("Current / original leg lengths:", np.round(self.lk,3), "/", np.round(self.l0,3))
        if self.lk >= self.l0:
            print("  Lift-off! theta:", self.theta)
            print("  Lift-off coordinates (x,y):", self.x, self.y)
            return True
        else:
            return False
    
    def check_fall(self):
        """
        TODO: check if robot has fallen / tripped
        """
        
        return

    
    
    def save_state(self, array):
        """ Access and save state information to array """
        array.append([self.x, self.xdot, self.y, self.ydot, self.theta, self.thetadot, self.lk, self.ldot])    
        return




def plot_animation(xvalues, yvalues, xlabel, ylabel):
    """
    Helper function to animate the display of yvalues over xvalues
    xlabel, ylabel are string names for axes. 
    Assumes the two values have the same length
    See more: https://www.geeksforgeeks.org/how-to-create-animations-in-python/
    """
    x = []
    y = []
    max_x = max(xvalues)
    min_x = min(xvalues)
    max_y = max(yvalues)
    min_y = min(yvalues)
    if len(xvalues) != len(yvalues):
        print("Error: two data arrays' lengths different")
        return 
    for i in range(len(xvalues)):
        x.append(xvalues[i])   # time axis
        y.append(yvalues[i])
        # Mention x and y limits to define their range
        plt.xlim(min_x, 1.1*max_x)
        plt.ylim(min_y, 1.1*max_y)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.plot(x, y, color = 'green')
        plt.pause(0.01)
        plt.show()
    return



"""---------------- IMPLEMENTATION ----------------"""

dT = 0.005  # seconds, maybe just Euler integrate
state_history = []  # store trajectory

# Initialize Pogo, propagate free flight
Pogo = Pogo_robot(100,1,0.3,0, 0.2, 0.5, -0.3, 2, 0)   # Initialize k,l,m and state
test_time = 150   # total number of timesteps to simulate

time = 0
state = 0   # flag 0 - free fall; 1 - contact
while time < test_time:
    if state == 0: 
        if Pogo.is_in_contact() == False:
            Pogo.flight() 
        else:                      
            # first contact  
            print("Contacted ground at", time, "step.")
            state = 1
            Pogo.first_contact()   # compute conversion only once
            Pogo.contact()         # do compression of leg
    else:   
        # state = 1
        if Pogo.takeoff_check() == False:
            Pogo.contact()
        else:
            print("Left the ground at", time, "step.")
            state = 0
            Pogo.flight()  
    
    Pogo.save_state(state_history)
    time += 1
    
    
print(Pogo.is_in_contact())

# Check for ground contact
# 

# Plot trajectories
x_history = []
for i in range(test_time):
    x_history.append(state_history[i][0])

plt.plot(x_history)
plt.xlabel("Timestep")
plt.ylabel("x")
plt.show()

y_history = []
for i in range(test_time):
    y_history.append(state_history[i][2])

plt.plot(y_history)
#plt.ylim(0, 0.6)
#plt.xlim(0, 40)
plt.xlabel("Timestep")
plt.ylabel("y")
plt.show()

theta_history = []
for i in range(test_time):
    theta_history.append(state_history[i][4])

plt.plot(theta_history)
plt.xlabel("Timestep")
plt.ylabel("theta")
plt.show()

lk_history = []
for i in range(test_time):
    lk_history.append(state_history[i][6])
    

plt.plot(lk_history)
plt.xlabel("Timestep")
plt.ylabel("Leg length")
plt.show()


plt.plot(x_history, y_history)

# Plot animation testing
#plot_animation(x_history, y_history, "x coordinate","y coordinate")


#values = []
#for i in range(50):
#    values.append(i)
#plot_animation(values, "Testing")
    
    


