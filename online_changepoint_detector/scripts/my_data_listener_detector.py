#!/usr/bin/env python

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

import rospy
import numpy as np
import cv2
from cv_bridge import CvBridge

# import StudentTMulti as st
import StudentT as st
import Detector as dt
import hazards as hz

from std_msgs.msg import Header
from jsk_recognition_msgs.msg import Spectrum, Histogram
from sensor_msgs.msg import Image
from functools import partial
import bayesian_changepoint_detection.online_changepoint_detection as oncd
import bayesian_changepoint_detection.offline_changepoint_detection as offcd
from std_srvs.srv import Trigger, TriggerResponse


class Data_listener(object):
    def __init__(self):
        # Get spectrum length
        spectrum_msg = rospy.wait_for_message('~spectrum',
                                              Spectrum)
        self.dim = len(spectrum_msg.amplitude)
        self.init_work_variables()
        self.sub = rospy.Subscriber("~spectrum",
                                    Spectrum, self.cb, queue_size=3, tcp_nodelay=True)
        self.data_selection_type = rospy.get_param("~data_selection_type", "average")
        self.rate = rospy.get_param("~rate", 20)
        self.timestamp = None
        rospy.Service("~start_logging",
                      Trigger, self.start_cb)
        rospy.Service("~stop_logging",
                      Trigger, self.stop_cb)
        self.pub_img = rospy.Publisher(
            '~output', Image, queue_size=1)
        self.pub_result = rospy.Publisher(
            '~result', Histogram, queue_size=1)
        self.threshold = rospy.get_param('~threshold', 0.6)

    def start_cb(self, req):
        self.data = []
        self.is_logging = True
        return TriggerResponse(success=True)
    def stop_cb(self, req):
        self.is_logging = False
        if len(self.data) == 0:
            return TriggerResponse(success=False,
                                   message="data length zero")
        self.data = np.array(self.data)
        Q, P, Pcp = offcd.offline_changepoint_detection(
            self.data,
            partial(offcd.const_prior, l=(len(self.data)+1)),
            offcd.gaussian_obs_log_likelihood,
            truncate=-40
        )
        now = rospy.Time.now()
        fig, ax = plt.subplots(2, 1, figsize=[18, 16], sharex='all')
        ax[0].plot(self.data[:])
        ax[1].plot(np.exp(Pcp).sum(0))
        ax[1].plot([0, len(Pcp)], [self.threshold, self.threshold])
        cps = np.where(np.exp(Pcp).sum(0) > self.threshold)
        cps = np.concatenate([[0], cps[0], [len(self.data)+1]])
        aves = []
        for before, after in zip(cps[:-1], cps[1:]):
            aves.append(np.average(self.data[before:after]))
        self.pub_result.publish(Histogram(
            header=Header(stamp=now),
            histogram=aves))
        print(cps[1:-1], aves)
        # plt.pause(.001)
        try:
            filename = '/tmp/%s.png'%now
            plt.savefig(filename)
            img = cv2.imread(filename)
            img_msg = CvBridge().cv2_to_imgmsg(img, 'bgr8')
            img_msg.header.stamp = now
            self.pub_img.publish(img_msg)
        finally:
            plt.close()
        return TriggerResponse(success=True)

    def init_work_variables(self):
        self.X = np.array([0.0] * self.dim, dtype=np.float32)
	# self.prior = st.StudentTMulti(self.dim)
	self.prior = st.StudentT(0.1, .01, 1, 0)
	self.detector = dt.Detector()
        self.is_logging = False
        self.data = []

    def stop(self):
        '''Stop the object'''
	
	# maxes, CP, theta = self.detector.retrieve(self.prior)
  	# rospy.loginfo("\nChangepoints locations: %f\n", CP)
  	# rospy.loginfo("\nSegment parameters: %f\n", theta)
        pass

    def cb(self, msg):
        if self.timestamp is not None and \
           msg.header.stamp - self.timestamp < rospy.Duration(1./self.rate):
            return
        self.timestamp = msg.header.stamp
        spectrum = np.array(msg.amplitude, dtype=np.float32)
        freq = np.array(msg.frequency, dtype=np.float32)
        ave = np.sum(freq * spectrum) / np.sum(spectrum)
        # self.X = msg.amplitude
        if self.data_selection_type == "average":
            self.X = ave
        elif self.data_selection_type == "max":
            self.X = freq[np.argmax(spectrum)]
        else:
            rospy.logerr("no such data_selection_type: %s"%self.data_selection_type)
            return

	# self.detector.detect(self.X,partial(hz.constant_hazard,lam=200),self.prior)
	# self.detector.detect(self.X,
        #                      partial(hz.constant_hazard,lam=250),
        #                      self.prior)
    	# self.detector.plot_data_CP(self.X)

        if self.is_logging:
            self.data.append(self.X)

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
