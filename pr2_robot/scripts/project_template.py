#!/usr/bin/env python

# Import modules
import numpy as np
import sklearn
from sklearn.preprocessing import LabelEncoder
import pickle
from sensor_stick.srv import GetNormals
from sensor_stick.features import compute_color_histograms
from sensor_stick.features import compute_normal_histograms
from visualization_msgs.msg import Marker
from sensor_stick.marker_tools import *
from sensor_stick.msg import DetectedObjectsArray
from sensor_stick.msg import DetectedObject
from sensor_stick.pcl_helper import *

import rospy
import tf
from geometry_msgs.msg import Pose
from std_msgs.msg import Float64
from std_msgs.msg import Int32
from std_msgs.msg import String
from pr2_robot.srv import *
from rospy_message_converter import message_converter
import yaml

TEST_NUM = 3

# Helper function to get surface normals
def get_normals(cloud):
    get_normals_prox = rospy.ServiceProxy('/feature_extractor/get_normals', GetNormals)
    return get_normals_prox(cloud).cluster

# Helper function to create a yaml friendly dictionary from ROS messages
def make_yaml_dict(test_scene_num, arm_name, object_name, pick_pose, place_pose):
    yaml_dict = {}
    yaml_dict["test_scene_num"] = test_scene_num.data
    yaml_dict["arm_name"]  = arm_name.data
    yaml_dict["object_name"] = object_name.data
    yaml_dict["pick_pose"] = message_converter.convert_ros_message_to_dictionary(pick_pose)
    yaml_dict["place_pose"] = message_converter.convert_ros_message_to_dictionary(place_pose)
    return yaml_dict

# Helper function to output to yaml file
def send_to_yaml(yaml_filename, dict_list):
    data_dict = {"object_list": dict_list}
    with open(yaml_filename, 'w') as outfile:
        yaml.dump(data_dict, outfile, default_flow_style=False)

def make_statistical_outlier_filter(points):
    outlier_filter = points.make_statistical_outlier_filter()
    outlier_filter.set_mean_k(15)
    outlier_filter.set_std_dev_mul_thresh(0.1)
    
    return outlier_filter.filter()

def voxel_grid_downsampling(points):
    vox = points.make_voxel_grid_filter()
    LEAF_SIZE = 0.006
    vox.set_leaf_size(LEAF_SIZE, LEAF_SIZE, LEAF_SIZE)
    
    return vox.filter()

def pass_through_filter(points, filter_axis, axis_min, axis_max):
    passthrough_filter = points.make_passthrough_filter()
    passthrough_filter.set_filter_field_name(filter_axis)
    passthrough_filter.set_filter_limits(axis_min, axis_max)

    return passthrough_filter.filter()

def ransac_filter(points):
    seg = points.make_segmenter()

    seg.set_model_type(pcl.SACMODEL_PLANE)
    seg.set_method_type(pcl.SAC_RANSAC)

    max_distance = 0.01
    seg.set_distance_threshold(max_distance)

    inliers, coefficients = seg.segment()

    return inliers

def euclidean_clustering(points):
    white_cloud = XYZRGB_to_XYZ(points)
    tree = white_cloud.make_kdtree()

    ec = white_cloud.make_EuclideanClusterExtraction()
    ec.set_ClusterTolerance(0.06)
    ec.set_MinClusterSize(60)
    ec.set_MaxClusterSize(25000)
    ec.set_SearchMethod(tree)

    cluster_indices = ec.Extract()

    return cluster_indices, white_cloud

def cluster_mask_point_cloud(cluster_indices, white_cloud):
    cluster_color = get_color_list(len(cluster_indices))

    color_cluster_point_list = []

    for j, indices in enumerate(cluster_indices):
        for i, indice in enumerate(indices):
            color_cluster_point_list.append([white_cloud[indice][0],
                                             white_cloud[indice][1],
                                             white_cloud[indice][2],
                                    rgb_to_float(cluster_color[j])])

    #Create new cloud containing all clusters, each with unique color
    cluster_cloud = pcl.PointCloud_PointXYZRGB()
    cluster_cloud.from_list(color_cluster_point_list)

    return cluster_cloud

# Callback function for your Point Cloud Subscriber
def pcl_callback(pcl_msg):

# Exercise-1 TODOs:

    # TODO: Convert ROS msg to PCL data
    pcl_data = ros_to_pcl(pcl_msg)
    
    # TODO: Statistical Outlier Filtering
    points = make_statistical_outlier_filter(pcl_data)

    # TODO: Voxel Grid Downsampling
    points = voxel_grid_downsampling(points)

    # TODO: PassThrough Filter
    points = pass_through_filter(points, 'y', -0.4, 0.5)
    points = pass_through_filter(points, 'z', 0.6, 1.1)

    # TODO: RANSAC Plane Segmentation
    inliers = ransac_filter(points)

    # TODO: Extract inliers and outliers
    extracted_inliers = points.extract(inliers, negative=False)
    extracted_outliers = points.extract(inliers, negative=True)

# Exercise-2 TODOs:

    # TODO: Euclidean Clustering
    cluster_indices, white_cloud = euclidean_clustering(extracted_outliers)

    # TODO: Create Cluster-Mask Point Cloud to visualize each cluster separately
    cluster_cloud = cluster_mask_point_cloud(cluster_indices, white_cloud)

    # TODO: Convert PCL data to ROS messages
    ros_cloud_table = pcl_to_ros(extracted_inliers)
    ros_cloud_objects = pcl_to_ros(extracted_outliers)
    ros_cluster_cloud = pcl_to_ros(cluster_cloud)

    # TODO: Publish ROS messages
    pcl_objects_pub.publish(ros_cloud_objects)
    pcl_table_pub.publish(ros_cloud_table)
    pcl_cluster_pub.publish(ros_cluster_cloud)

# Exercise-3 TODOs:

    # Classify the clusters! (loop through each detected cluster one at a time)
    detected_objects_labels = []
    detected_objects = []

    for index, pts_list in enumerate(cluster_indices):
        # Grab the points for the cluster from the extracted outliers (cloud_objects)
        pcl_cluster = extracted_outliers.extract(pts_list)
        # TODO: convert the cluster from pcl to ROS using helper function
        ros_cluster = pcl_to_ros(pcl_cluster)

        # Extract histogram features
        # TODO: complete this step just as is covered in capture_features.py
        chists = compute_color_histograms(ros_cluster, using_hsv=True)
        normals = get_normals(ros_cluster)
        nhists = compute_normal_histograms(normals)
        feature = np.concatenate((chists, nhists))

        # Make the prediction, retrieve the label for the result
        # and add it to detected_objects_labels list
        prediction = clf.predict(scaler.transform(feature.reshape(1,-1)))
        label = encoder.inverse_transform(prediction)[0]
        detected_objects_labels.append(label)

        # Publish a label into RViz
        label_pos = list(white_cloud[pts_list[0]])
        label_pos[2] += .4
        object_markers_pub.publish(make_label(label,label_pos, index))

        # Add the detected object to the list of detected objects.
        do = DetectedObject()
        do.label = label
        do.cloud = ros_cluster
        detected_objects.append(do)
        
    # Publish the list of detected objects
    rospy.loginfo('Detected {} objects: {}'.format(len(detected_objects_labels), detected_objects_labels))

    # Publish the list of detected objects
    # This is the output you'll need to complete the upcoming project!
    detected_objects_pub.publish(detected_objects)

    # Suggested location for where to invoke your pr2_mover() function within pcl_callback()
    # Could add some logic to determine whether or not your object detections are robust
    # before calling pr2_mover()
    detected_objects_list = detected_objects

    try:
        pr2_mover(detected_objects_list)
    except rospy.ROSInterruptException:
        pass

# function to load parameters and request PickPlace service
def pr2_mover(object_list):
    # TODO: Initialize variables
    objects = {}
    dropboxes = {}
    dict_list = []

    # TODO: Get/Read parameters
    object_list_param = rospy.get_param('/object_list')
    dropbox_param = rospy.get_param('/dropbox')

    # TODO: Parse parameters into individual variables
    for obj in object_list_param:
        objects[obj['name']] = obj['group']

    for dropbox in dropbox_param:
        dropboxes[dropbox['group']] = (dropbox['name'], dropbox['position'])

    # TODO: Rotate PR2 in place to capture side tables for the collision map

    # TODO: Loop through the pick list
    for obj in object_list:
        # TODO: Get the PointCloud for a given object and obtain it's centroid
        points_arr = ros_to_pcl(obj.cloud).to_array()
        centroid=np.mean(points_arr, axis=0)[:3]

        PICK_POSE = Pose()
        PICK_POSE.position.x = float(centroid[0])
        PICK_POSE.position.y = float(centroid[1])
        PICK_POSE.position.z = float(centroid[2])

        # TODO: Create 'place_pose' for the object
        group = objects[obj.label]
        dropbox_name, dropbox_pos = dropboxes[group]

        PLACE_POSE = Pose()
        PLACE_POSE.position.x = float(dropbox_pos[0])
        PLACE_POSE.position.y = float(dropbox_pos[1])
        PLACE_POSE.position.z = float(dropbox_pos[2])

        # TODO: Assign the arm to be used for pick_place
        WHICH_ARM = String()
        WHICH_ARM.data = dropbox_name

        # TODO: Create a list of dictionaries (made with make_yaml_dict()) for later output to yaml format
        TEST_SCENE_NUM = Int32()
        TEST_SCENE_NUM.data = TEST_NUM
        OBJECT_NAME = String()
        OBJECT_NAME.data = str(obj.label)

        yaml_dict = make_yaml_dict(TEST_SCENE_NUM, WHICH_ARM, OBJECT_NAME, PICK_POSE, PLACE_POSE)
        dict_list.append(yaml_dict)    

        # Wait for 'pick_place_routine' service to come up
        # rospy.wait_for_service('pick_place_routine')

        try:
            pick_place_routine = rospy.ServiceProxy('pick_place_routine', PickPlace)

            # TODO: Insert your message variables to be sent as a service request
            resp = pick_place_routine(TEST_SCENE_NUM, OBJECT_NAME, WHICH_ARM, PICK_POSE, PLACE_POSE)

            print ("Response: ",resp.success)

        except rospy.ServiceException, e:
            print "Service call failed: %s"%e

    # TODO: Output your request parameters into output yaml file
    send_to_yaml('/home/robond/catkin_ws/src/RoboND-Perception-Project/pr2_robot/scripts/output_{}.yaml'.format(TEST_NUM), dict_list)


if __name__ == '__main__':

    # TODO: ROS node initialization
    rospy.init_node('clustering', anonymous=True)

    # TODO: Create Subscribers
    pcl_sub = rospy.Subscriber("/pr2/world/points", pc2.PointCloud2, pcl_callback, queue_size=1)

    # TODO: Create Publishers
    pcl_objects_pub = rospy.Publisher("/pcl_objects", PointCloud2, queue_size=1)
    pcl_table_pub = rospy.Publisher("/pcl_table", PointCloud2, queue_size=1)
    pcl_cluster_pub = rospy.Publisher("/pcl_cluster", PointCloud2, queue_size=1)

    object_markers_pub = rospy.Publisher("/object_markers", Marker, queue_size=1)
    detected_objects_pub = rospy.Publisher("/detected_objects", DetectedObjectsArray, queue_size=1)

    # TODO: Load Model From disk
    model_path = '/home/robond/catkin_ws/src/RoboND-Perception-Project/pr2_robot/scripts/model{}/'.format(TEST_NUM)
    model = pickle.load(open(model_path + 'model.sav', 'rb'))
    clf = model['classifier']
    encoder = LabelEncoder()
    encoder.classes_ = model['classes']
    scaler = model['scaler']

    # Initialize color_list
    get_color_list.color_list = []

    # TODO: Spin while node is not shutdown
    while not rospy.is_shutdown():
  	rospy.spin()
