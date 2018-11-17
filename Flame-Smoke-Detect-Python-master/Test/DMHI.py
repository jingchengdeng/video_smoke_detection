# *^_^* coding:utf-8 *^_^*
"""
利用LK光流追踪测特征点，对特征点生成MHI，DMHI没有实现。缺陷：针对特征点不能检测目标轮廓
"""

__author__ = 'stone'
__date__ = '16-1-5'

import cv2
import numpy as np

MHI_DURATION = 5
DEFAULT_THRESHOLD = 35
MAX_TIME_DELTA = 0.25
MIN_TIME_DELTA = 0.05
ZOOM_TIME = 1


def zoom_down(frames, n):
    """
    zoom down/up the image
    """
    h, w, r = frames.shape  # h:height w:width r:ret
    small_frames = cv2.resize(frames, (w / n, h / n), interpolation=cv2.INTER_CUBIC)
    return small_frames


def clock():
    return cv2.getTickCount() / cv2.getTickFrequency()


def draw_flow(img, flow, step=16):
    h, w = img.shape[:2]
    y, x = np.mgrid[step / 2:h:step, step / 2:w:step].reshape(2, -1).astype(int)
    fx, fy = flow[y, x].T
    lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.polylines(vis, lines, 0, (0, 255, 0))
    for (x1, y1), (x2, y2) in lines:
        cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
    return vis


if __name__ == '__main__':
    import sys

    # 定义特征点和LK光流法参数
    feature_params = dict(
        maxCorners=300,
        qualityLevel=0.3,
        minDistance=7,
        blockSize=200
    )

    lk_params = dict(
        winSize=(15, 15),
        maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
    )

    vName = '../medias/videos/side/daria_side.avi'
    # CTC_FG.028_9.mpg
    # Homewood_BGsmokey.050_10.mpg
    # Heavenly_FG.052_09.mpg
    # CTC.BG.055_11.mpg
    # camera2.mov
    # 3_2012-07-17_15-15-44.avi
    # ../medias/videos/side/daria_side.avi

    try:
        video_src = sys.argv[1]
    except:
        video_src = vName

    capture = cv2.VideoCapture(video_src)
    ret, prev_frame = capture.read()
    prev_frame = zoom_down(prev_frame, ZOOM_TIME)
    h, w = prev_frame.shape[:2]
    motion_history = np.zeros((h, w), np.float32)

    gray_prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    # 取特征点
    p0 = cv2.goodFeaturesToTrack(gray_prev_frame, mask=None, **feature_params)

    while True:
        ret, frame = capture.read()
        if frame is None:
            print "video finished."
            break
        frame = zoom_down(frame, ZOOM_TIME)
        h, w = frame.shape[:2]
        motion_silhouette = np.zeros((h, w), np.uint8)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 金字塔Lucas-Kanade光流法
        p1, st, err = cv2.calcOpticalFlowPyrLK(gray_prev_frame, gray_frame, p0, None, **lk_params)

        # 把特征点放大作为silhouette
        for circle_center in p1:
            motion_silhouette = cv2.circle(motion_silhouette, (circle_center[0][0], circle_center[0][1]), 2, 1, -1)

        timestamp = clock()

        # 构建mhi运动模板
        cv2.motempl.updateMotionHistory(motion_silhouette, motion_history, timestamp, MHI_DURATION)

        # 显示mhi图像
        vis = np.uint8(np.clip((motion_history - (timestamp - MHI_DURATION)) / MHI_DURATION, 0, 1) * 255)
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

        cv2.imshow('mhi', vis)
        cv2.imshow('frame', frame)

        prev_frame = frame.copy()
        k = cv2.waitKey(50) & 0xFF
        if k == 27:
            break

    cv2.destroyAllWindows()
    capture.release()
