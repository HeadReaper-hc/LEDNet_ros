#!/home/hc/anaconda3/bin/python
import os
import sys
import argparse
# from PIL import Image
import PIL
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

import torch
from torchvision import transforms

cur_path = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(cur_path, '..'))
from model.lednet import LEDNet
import utils as ptutil

import rospy
import cv2
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image

def parse_args():
    parser = argparse.ArgumentParser(description='Demo for LEDNet from a given image')

    parser.add_argument('--input-pic', type=str, default=os.path.join(cur_path, 'png/000000.png'),
                        help='path to the input picture')
    parser.add_argument('--pretrained', type=str,
                        default=os.path.expanduser('/home/hc/LEDNet-master/LEDNet_final.pth'),
                        help='Default Pre-trained model root.')
    parser.add_argument('--cuda', type=ptutil.str2bool, default='true',
                        help='demo with GPU')

    opt = parser.parse_args()
    return opt

def cv2PIL(img, colorChange):
    image = PIL.Image.fromarray(cv2.cvtColor(img, colorChange))
    return image

def PIL2cv(img, colorChange):
    image = cv2.cvtColor(np.asarray(img), colorChange)
    return image

class semantic:

  def __init__(self, image_topic, device, pretrained):
    self.image_pub = rospy.Publisher("semantic_img",Image,queue_size=10)

    self.bridge = CvBridge()
    self.image_sub = rospy.Subscriber(image_topic,Image,self.callback,queue_size=1)

    self.transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    self.device = device
    self.model = LEDNet(19).to(device)
    self.model.load_state_dict(torch.load(pretrained))
    self.model.eval()

  def callback(self,data):
    try:
      cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
      print(e)

    pilImg = cv2PIL(cv_image,cv2.COLOR_BGR2RGB)
    img = self.transform(pilImg).unsqueeze(0).to(self.device)
    with torch.no_grad():
        output = self.model(img)
    predict = torch.argmax(output, 1).squeeze(0).cpu().numpy()
    mask = ptutil.get_color_pallete(predict, 'citys')
    mask.save(os.path.join(cur_path, 'png/output.png'))
    mmask = cv2.imread(os.path.join(cur_path, 'png/output.png'))
    # plt.imshow(mmask)
    # plt.show()
    # cv2.imshow("OpenCV",mmask)
    # cv2.waitKey(1)

    try:
      self.image_pub.publish(self.bridge.cv2_to_imgmsg(mmask, "bgr8"))
    except CvBridgeError as e:
      print(e)




if __name__ == '__main__':
    args = parse_args()
    device = torch.device('cpu')
    if args.cuda:
        device = torch.device('cuda')
    # Load Model

    #Ros
    image_topic = '/cam0/image_raw'
    ic = semantic(image_topic, device, args.pretrained)
    rospy.init_node('orb_semantic')

    rospy.spin()
