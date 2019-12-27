#!/usr/bin/env python
import rospy
import numpy as np
import StudentTMulti as st
import Detector as dt
import hazards as hz
import matplotlib.pyplot as plt

from jsk_recognition_msgs.msg import Spectrum
from functools import partial


class Data_listener(object):
    def __init__(self):
        # Get spectrum length
        spectrum_msg = rospy.wait_for_message('~spectrum', Spectrum)

        self.dim = len(spectrum_msg.amplitude)
        self.init_work_variables()
        self.sub = rospy.Subscriber("~spectrum", Spectrum, self.cb, queue_size=3, tcp_nodelay=True)

    def init_work_variables(self):
        self.X = np.array([0.0] * self.dim, dtype=np.float32)
	self.prior = st.StudentTMulti(self.dim)
	self.detector = dt.Detector()

    def stop(self):
        '''Stop the object'''
	
	maxes, CP, theta = self.detector.retrieve(self.prior)
  	rospy.loginfo("\nChangepoints locations: %f\n", CP)
  	rospy.loginfo("\nSegment parameters: %f\n", theta)

        self.pose_sub.unregister()
        self.ft_sub.unregister()

    def cb(self, msg):
        spectrum = np.array(msg.amplitude, dtype=np.float32)
        self.X = spectrum

	self.detector.detect(self.X,partial(hz.constant_hazard,lam=200),self.prior)
    	self.detector.plot_data_CP(self.X)

def main():
        rospy.init_node('data_listener', anonymous=True)
        rospy.loginfo("%s: Starting" % (rospy.get_name()))

	plt.ion()
        listen = Data_listener()

        rospy.spin()
        rospy.on_shutdown(close(listen))

def close(listen):
        rospy.loginfo("%s: Exiting" % (rospy.get_name()))
        listen.stop()


if __name__ == '__main__':
    main()
