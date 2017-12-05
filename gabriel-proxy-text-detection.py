#!/usr/bin/env python

import multiprocessing
import Queue
from optparse import OptionParser
import os
import pprint
import struct
import sys
import time
import pdb

dir_file = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(dir_file, "../.."))
import gabriel
import gabriel.proxy
import json
import cv2
import numpy as np
import base64
import subprocess
from PIL import Image
import pytesseract

LOG = gabriel.logging.getLogger(__name__)
ANDROID_CLIENT = True


def process_command_line(argv):
    VERSION = 'gabriel proxy : %s' % gabriel.Const.VERSION
    DESCRIPTION = "Gabriel cognitive assistance"

    parser = OptionParser(usage='%prog [option]', version=VERSION,
                          description=DESCRIPTION)

    parser.add_option(
        '-s', '--address', action='store', dest='address',
        help="(IP address:port number) of directory server")
    settings, args = parser.parse_args(argv)
    if len(args) >= 1:
        parser.error("invalid arguement")

    if hasattr(settings, 'address') and settings.address is not None:
        if settings.address.find(":") == -1:
            parser.error("Need address and port. Ex) 10.0.0.1:8081")
    return settings, args


class DummyVideoApp(gabriel.proxy.CognitiveProcessThread):

    word_list = set()
    word_list.add("flood")
    word_list.add("flight")
    word_list.add("flights")
    word_list.add("floodway")
    word_list.add("way")
    word_list.add("where")
    word_list.add("customs")
    word_list.add("custom")
    word_list.add("arrival")
    word_list.add("departure")
    word_list.add("stop")
    word_list.add("baggage")
    word_list.add("gate")
    word_list.add("noentry")
    word_list.add("no")
    word_list.add("entry")
    word_list.add("money")
    word_list.add("exchange")
    word_list.add("information")
    word_list.add("check")
    word_list.add("in")
    word_list.add("checkin")
    word_list.add("passport")
    word_list.add("toilet")
    word_list.add("gate")
    word_list.add("speed")
    word_list.add("limit")
    word_list.add("speedlimit")
    word_list.add("top")
    word_list.add("pop")
    word_list.add("mop")


    def word_in_dict(self,txt):
	splits = txt.split()
	final= ""
	for word in splits:
	    word = word.lower()
	    if word in self.word_list:
		final +=word
	return final
    def find_text(self, img):
        try:
            vis = img.copy()

            # Extract channels to be processed individually
            channels = cv2.text.computeNMChannels(img)
            # Append negative channels to detect ER- (bright regions over dark background)
            cn = len(channels) - 1

            for c in range(0, cn):
                channels.append((255 - channels[c]))
            # print len(channels)
            for i, channel in enumerate(channels):
                # if(i>3):
                #    continue;
                erc1 = cv2.text.loadClassifierNM1('trained_classifierNM1.xml')
                er1 = cv2.text.createERFilterNM1(erc1, 16, 0.00015, 0.13, 0.2, True, 0.1)

                erc2 = cv2.text.loadClassifierNM2('trained_classifierNM2.xml')
                er2 = cv2.text.createERFilterNM2(erc2, 0.5)

                regions = cv2.text.detectRegions(channel, er1, er2)
                rects = cv2.text.erGrouping(img, channel, [r.tolist() for r in regions])
                # rects = cv2.text.erGrouping(img,channel,[x.tolist() for x in regions], cv2.text.ERGROUPING_ORIENTATION_ANY,'../../GSoC2014/opencv_contrib/modules/text/samples/trained_classifier_erGrouping.xml',0.5)

                # Visualization
                for r in range(0, np.shape(rects)[0]):
                    rect = rects[r]

                    imCrop = vis[int(rect[1]):int(rect[1] + rect[3]), int(rect[0]):int(rect[0] + rect[2])]
                    # imCrop = vis[int(rect[1]):int(rect[3] - rect[1]), int(rect[0]):int(rect[2] - rect[0] )]
                    # cv2.imshow('imCrop', imCrop)
                    tesImg = Image.fromarray(imCrop)
                    txt = pytesseract.image_to_string(tesImg)
   		    print("tesseract text: "+ txt)
                    txt = self.word_in_dict(txt)
		    if (len(txt.strip()) != 0):
   		         print("tesseract text: "+ txt)
		         # print("apertium path : " +" echo \" "+ txt+ " \"| apertium -d /apertium-en-es en-es" )
		         pipe = subprocess.Popen(" echo \" "+ txt+ " \"| apertium -d /apertium-en-es en-es", shell=True,
                                                 stdout=subprocess.PIPE).stdout
                         convertedTxt = pipe.read()
			 covertedTxt = convertedTxt.translate(None,"*?")
			 # covertedTxt = convertedTxt.replace("?","");
                         print("found text:" + convertedTxt)
                         cv2.rectangle(vis, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (255, 0, 255), 1)
			 sub_face = vis[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]]
	                 # apply a gaussian blur on this new recangle image
                    	 sub_face = cv2.GaussianBlur(sub_face, (23, 23), 30)
                    	 # merge this blurry rectangle to our final image
                    	 vis[rect[1]:rect[1] + sub_face.shape[0], rect[0]:rect[0] + sub_face.shape[1]] = sub_face
	
        #            cv2.putText(vis, "wazza", (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 1,
        #                        (255, 255, 255), 2, 2)

                         cv2.putText(vis, convertedTxt, (rect[0], rect[1] + rect[3]), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (255, 255, 255), 2, 2)
                    return vis
        except Exception as e:
            print("find_text exception: "+ str(e))
            return img

    def add_to_byte_array(self, byte_array, extra_bytes):
        return struct.pack("!{}s{}s".format(len(byte_array), len(extra_bytes)), byte_array, extra_bytes)

    def handle(self, header, data):
        # PERFORM Cognitive Assistance Processing
        LOG.info("processing: ")
        LOG.info("%s\n" % header)
        np_data = np.fromstring(data, dtype=np.uint8)
        bgr_img = cv2.imdecode(np_data, cv2.IMREAD_COLOR)
        translated_img = self.find_text(bgr_img)
        # place code here
        if translated_img is not None:
            _, jpeg_img = cv2.imencode('.jpg', translated_img)

            if ANDROID_CLIENT:
                # old version return
                if gabriel.Const.LEGACY_JSON_ONLY_RESULT:
                    jpeg_str = base64.b64encode(jpeg_img)
                    msg = {
                        gabriel.Protocol_result.JSON_KEY_STATUS: 'success',
                        gabriel.Protocol_result.JSON_KEY_IMAGE: jpeg_str,
                        gabriel.Protocol_result.JSON_KEY_IMAGES_ANIMATION: [jpeg_str],
                        gabriel.Protocol_result.JSON_KEY_SPEECH: "mirror"
                    }
                    return json.dumps(msg)
                # new gabriel version return
                else:
                    # image data
                    header[gabriel.Protocol_result.JSON_KEY_STATUS] = 'success'
                    # numpy tostring is equal to tobytes
                    rtn_data = jpeg_img.tostring()
                    # header has (offset, size) for each data type
                    header[gabriel.Protocol_result.JSON_KEY_IMAGE] = (0, len(rtn_data))

                    # animation data
                    # animation is two images: before mirror and after mirror
                    offset = len(rtn_data)
                    _, ori_img = cv2.imencode('.jpg', bgr_img)
                    animation_data = [jpeg_img.tostring(), ori_img.tostring()]
                    # animation bytes format:
                    # first integer: number of frames
                    # for each frame: size of the frame + data
                    animation_bytes = struct.pack("!I", len(animation_data))
                    for frame in animation_data:
                        frame_bytes = struct.pack("!I{}s".format(len(frame)), len(frame), frame)
                        animation_bytes = self.add_to_byte_array(animation_bytes, frame_bytes)
                    rtn_data = self.add_to_byte_array(rtn_data, animation_bytes)
                    header[gabriel.Protocol_result.JSON_KEY_IMAGES_ANIMATION] = (offset, len(animation_bytes))

                    # speech data
                    offset = len(rtn_data)
                    speech = "mirror"
                    rtn_data = self.add_to_byte_array(rtn_data, speech)
                    header[gabriel.Protocol_result.JSON_KEY_SPEECH] = (offset, len(speech))
                    return rtn_data
            else:
                # python client can only handle image data
                return jpeg_img.tostring()
        else:
            print "some error occured, sending original image"
            _, jpeg_img = cv2.imencode('.jpg', bgr_img)
            return jpeg_img.tostring()


if __name__ == "__main__":
    result_queue = multiprocessing.Queue()
    print result_queue._reader

    settings, args = process_command_line(sys.argv[1:])
    ip_addr, port = gabriel.network.get_registry_server_address(settings.address)
    service_list = gabriel.network.get_service_list(ip_addr, port)
    LOG.info("Gabriel Server :")
    LOG.info(pprint.pformat(service_list))

    video_ip = service_list.get(gabriel.ServiceMeta.VIDEO_TCP_STREAMING_IP)
    video_port = service_list.get(gabriel.ServiceMeta.VIDEO_TCP_STREAMING_PORT)
    ucomm_ip = service_list.get(gabriel.ServiceMeta.UCOMM_SERVER_IP)
    ucomm_port = service_list.get(gabriel.ServiceMeta.UCOMM_SERVER_PORT)

    # image receiving and processing threads
    image_queue = Queue.Queue(gabriel.Const.APP_LEVEL_TOKEN_SIZE)
    print "TOKEN SIZE OF OFFLOADING ENGINE: %d" % gabriel.Const.APP_LEVEL_TOKEN_SIZE  # TODO
    video_receive_client = gabriel.proxy.SensorReceiveClient((video_ip, video_port), image_queue)
    video_receive_client.start()
    video_receive_client.isDaemon = True
    dummy_video_app = DummyVideoApp(image_queue, result_queue,
                                    engine_id='textDetection')  # dummy app for image processing
    dummy_video_app.start()
    dummy_video_app.isDaemon = True

    # result publish
    result_pub = gabriel.proxy.ResultPublishClient((ucomm_ip, ucomm_port), result_queue)
    result_pub.start()
    result_pub.isDaemon = True

    try:
        while True:
            time.sleep(1)
    except Exception as e:
        pass
    except KeyboardInterrupt as e:
        sys.stdout.write("user exits\n")
    finally:
        if video_receive_client is not None:
            video_receive_client.terminate()
        if dummy_video_app is not None:
            dummy_video_app.terminate()
        # if acc_client is not None:
        #    acc_client.terminate()
        # if acc_app is not None:
        #    acc_app.terminate()
        result_pub.terminate()
