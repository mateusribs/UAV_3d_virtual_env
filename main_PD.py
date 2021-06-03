import sys, os
import time

#Panda 3D Imports
from panda3d.core import Filename
from direct.showbase.ShowBase import ShowBase
from panda3d.core import loadPrcFile
loadPrcFile('./config/conf.prc')

from models.world_setup import world_setup, quad_setup
from computer_vision.cameras_setup import cameras
from environment.position2 import quad_sim


mydir = os.path.abspath(sys.path[0])

mydir = Filename.fromOsSpecific(mydir).getFullpath()

T = 0.01

class MyApp(ShowBase):
    def __init__(self):

        ShowBase.__init__(self)     
        render = self.render


        # MODELS SETUP
        world_setup(self, render, mydir)
        quad_setup(self, render, mydir)

        self.quad_sim = quad_sim(self)
        
        self.look_at = False
        self.last_time_press = 0
        
        

        self.taskMgr.add(self.controller_function, 'Controller Read')



    def controller_function(self, task):
        init_time = time.time()

        if task.frame==0:
            self.pos, self.vel, self.ang, self.ang_vel = self.quad_sim.quad_reset_random()
        else: 
            pos, vel, ang, ang_vel = self.quad_sim.step(self.pos, self.vel, self.ang, self.ang_vel, task.frame)
            self.pos = pos
            self.vel = vel
            self.ang = ang
            self.ang_vel = ang_vel
            print('Step:', task.frame)
        # while True:
        #     if time.time()-init_time >= T:
        #         return task.cont
        return task.cont
           

app = MyApp()

app.run()