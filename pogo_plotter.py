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
    def __init__(self, k, M, l0, init_x, init_xdot, init_y, init_ydot, \
                 init_theta, init_thetadot):
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
        
        self.foot_x = self.x + self.lk*np.sin(self.theta - np.pi/2)
        self.foot_y = self.y - self.lk*np.cos(self.theta - np.pi/2)
        
        self.theta0 = init_theta   # save initial angle; suppose leg goes back every time
        
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
        self.theta = self.theta0   # revert back to theta - but check swing oK?
        self.thetadot = 0    # no change
        self.lk = self.l0    # back to equilibrium point
        self.ldot = 0   
        
        self.foot_x = self.x + self.lk*np.sin(self.theta - np.pi/2)
        self.foot_y = self.y - self.lk*np.cos(self.theta - np.pi/2)
        
        
        
        return 
  
    def is_in_contact(self):
        """
        Flag function to compute if foot lands on the 
        ground (not takeoff check)
        """ 
        if self.foot_y <= 0:
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
            print(" ! IMPACT ERROR: velocity off")
            return
        
        # This check is good
        #print("  Check: impact speed", total_v, "post-impact speed", polar_coord_speed)
        
        return
    
    def contact(self):
        """Contacted mechanics with the ground"""
        prev_state = np.array([self.x, self.xdot, self.y, self.ydot, self.theta, self.thetadot, self.lk, self.ldot])    
    
        # l_ddot formula: -k/M(l_k - l_k0) - g sin(theta) + l*theta_dot^2 
        l_ddot = -self.k/self.M*(prev_state[6]-self.l0) - 9.81*np.sin(prev_state[4]) + prev_state[6]*(prev_state[5]**2)
        
        
        # theta_ddot formula: -2*l_dot/l*theta_dot - g/l cos(theta)
        theta_ddot = -2*prev_state[7]/prev_state[6]*prev_state[5] - 9.81/prev_state[6]*np.cos(prev_state[4])
        #print("    theta_ddot:",np.round(theta_ddot,3))
    
        # calculate l, theta 
        self.theta    += prev_state[5]*dT  # use previous step's accelerations
        self.thetadot += theta_ddot*dT
        self.lk       += prev_state[7]*dT
        self.ldot     += l_ddot*dT   
        
        #print("    theta:", np.round(self.theta,3))
        #print("    lk:", np.round(self.lk,3))
    
        # Update Cartesian velocities with both tangential and radial components        
        v_tan = abs(prev_state[6]*prev_state[5])          # tangential velocity, lk*thetadot
        total_v_polar = (v_tan**2 + self.ldot**2)**0.5    # for checking
        
        # Euler timestep forward 
        self.xdot = -v_tan*np.cos(prev_state[4] - np.pi/2) -\
                    self.ldot*np.cos(np.pi - prev_state[4])
        self.ydot = v_tan*np.sin(prev_state[4]-np.pi/2) - \
                    self.ldot*np.sin(np.pi - prev_state[4])
        #print("  Check xdot, ydot:", self.xdot, self.ydot)
        
        new_total_v = ((self.ydot)**2+(self.xdot)**2)**0.5   # for checking
        
        #self.foot_x = self.x + self.lk*np.sin(self.theta - np.pi/2)
        #self.foot_y = self.y - self.lk*np.cos(self.theta - np.pi/2)
        
        # force set this at 0
        self.foot_y = 0
        self.y = self.lk*np.cos(self.theta - np.pi/2)  # by geometry
        # don't change self.x
        self.x = self.foot_x - self.lk*np.sin(self.theta - np.pi/2)
        
        # Check velocity conversion makes sense
        if abs((new_total_v/total_v_polar)-1) > 0.01:
            print(" ! CONTACT ERROR: Cartesian to polar velocity off")
            return
        
        print("    foot_y", self.foot_y)
        
        #print("    body x:", np.round(self.x,3))
        #print("    body y:", np.round(self.y,3))
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
        if self.foot_y <= -0.05:            
            print("Max distance travelled:", self.x)
            return True
        return False

    
    
    def save_state(self, array):
        """ Access and save state information to array """
        array.append([self.x, self.xdot, self.y, self.ydot, self.theta, \
                      self.thetadot, self.lk, self.ldot, self.foot_x, \
                          self.foot_y])    
        return




def plot_animation(xvalues, yvalues, xvalues2, yvalues2, xlabel, ylabel):
    """
    Helper function to animate the display of yvalues over xvalues
    xlabel, ylabel are string names for axes. 
    Assumes the two values have the same length
    See more: https://www.geeksforgeeks.org/how-to-create-animations-in-python/
    """
    # Initialize holder lists
    x, x2, y, y2 = [], [], [], []
    
    max_x = max(max(xvalues), max(xvalues2))
    min_x = 0
    max_y = max(max(yvalues), max(yvalues2))
    min_y = 0 # should be this ideally
    #min_y = min(min(yvalues), min(yvalues2))
    if len(xvalues) != len(yvalues):
        print("Error: two data arrays' lengths different")
        return 
    for i in range(len(xvalues)):
        x.append(xvalues[i])   
        x2.append(xvalues2[i])
        y.append(yvalues[i])
        y2.append(yvalues2[i])
        # Mention x and y limits to define their range
        plt.xlim(min_x, 1.1*max_x)
        plt.ylim(min_y, 1.1*max_y)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.plot(x, y, color = 'blue')
        plt.plot(x2, y2, color = 'green')  # foot
        plt.plot(x[-1], y[-1], 'bo')
        plt.plot(x2[-1], y2[-1], 'go') 
        plt.pause(0.01)
        plt.show()
    return



"""---------------- IMPLEMENTATION ----------------"""

dT = 0.005  # seconds, maybe just Euler integrate
state_history = []  # store trajectory

# Initialize Pogo, propagate free flight
Pogo = Pogo_robot(200,1,0.3,0, 0.2, 0.5, -0.3, 2, 0)   # Initialize k,l,m and state
test_time = 100   # total number of timesteps to simulate

time = 0
state = 0   # flag 0 - free fall; 1 - contact
while time < test_time and Pogo.check_fall() == False:
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
    
if Pogo.check_fall() == True:
    print("Fell at step:", time)
    
    

# Check for ground contact
# 

# Plot trajectories
x_history = []
x_foot_history = []
for i in range(len(state_history)):
    x_history.append(state_history[i][0])
    x_foot_history.append(state_history[i][8])

plt.plot(x_history)
plt.plot(x_foot_history)
plt.xlabel("Timestep")
plt.ylabel("x")
plt.show()

y_history = []
y_foot_history = []
for i in range(len(state_history)):
    y_history.append(state_history[i][2])
    y_foot_history.append(state_history[i][9])

plt.plot(y_history)
plt.plot(y_foot_history)
#plt.ylim(0, 0.6)
#plt.xlim(0, 40)
plt.xlabel("Timestep")
plt.ylabel("y")
plt.show()

theta_history = []
for i in range(len(state_history)):
    theta_history.append(state_history[i][4])

plt.plot(theta_history)
plt.xlabel("Timestep")
plt.ylabel("theta")
plt.show()

lk_history = []
for i in range(len(state_history)):
    lk_history.append(state_history[i][6])
    

plt.plot(lk_history)
plt.xlabel("Timestep")
plt.ylabel("Leg length")
plt.show()


plt.plot(x_history, y_history, color="blue")
plt.plot(x_foot_history, y_foot_history, color="green")
plt.xlabel("x coordinate")
plt.ylabel("y coordinate")
plt.show()

# Plot animation testing
plot_animation(x_history, y_history, x_foot_history, y_foot_history, "x coordinate","y coordinate")
   
    


