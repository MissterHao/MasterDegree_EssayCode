
from skimage import img_as_ubyte
import json
import logging
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import requests
# from Image import ImageReader
from skimage import io
from skimage.filters import hessian, sato
from skimage.segmentation import (felzenszwalb, mark_boundaries, quickshift,
                                  slic, watershed)
from skimage.util import img_as_float
from sklearn.metrics import (auc, classification_report, confusion_matrix,
                             roc_auc_score, roc_curve)

from Voter import FelzenszwalbVoter, QuickShiftVoter, SLICVoter, WatershedVoter

target_dir = "image_data"
origin_dir = os.path.join(target_dir, "original")
label_dir = os.path.join(target_dir, "label")

origin_files = os.listdir(origin_dir)
label_files = os.listdir(label_dir)

START_X = 17
START_Y = 17
WIDTH = 850
HEIGHT = 850

END_X = START_X+WIDTH
END_Y = START_Y+HEIGHT


class AnalysisJob():

    def __init__(self, filename=None):
        if not filename:
            raise ValueError

        self.filename = filename

        img = cv2.imread(os.path.join(label_dir, filename), 0)
        if img is not None:
            self.labelImg = img.copy()[START_X:(
                WIDTH+START_X), START_X:(WIDTH+START_X)]

        img = cv2.imread(os.path.join(origin_dir, filename), 0)
        if img is not None:
            self.originalImg = img.copy()[START_X:(
                WIDTH+START_X), START_X:(WIDTH+START_X)]
            self._original = img.copy()

    def upload_weights(self, weights):
        # res = requests.get("http://localhost:8000/api/uploadWeight?weights={}".format(
            # ",".join(weights)
        # ))

        # jres = json.loads(res.text)
        # self.weights_id = jres["id"]
        self.weights = weights

        # return self.weights_id

    def getHistoryToken(self, note="") -> str:
        return
        res = requests.get("http://localhost:8000/api/token".format(
            ",".join(weights)
        ))

        jres = json.loads(res.text)
        self.token = jres["token"]

        return self.token

    # 前處理
    def preprocessing(self):
        img = self.originalImg
        types = {
            0: cv2.MORPH_RECT,
            1: cv2.MORPH_CROSS,
            2: cv2.MORPH_ELLIPSE,
        }
        ksize = 21
        ktype = 1
        B = cv2.getStructuringElement(types[ktype], (ksize, ksize))
        BTH = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, B)
        WTH = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, B)
        FEN = img + WTH - BTH

        # 保留40以下的
        img_under_40 = np.where(img < 40, img, 0)
        FEN = np.where(img < 40, img, FEN)
        FEN = np.where(img > 200, img, FEN)

        # =====================================================================================================================================
        # Contrast limited adaptive histogram equalization
        # 新增對比
        # CLAHE
        # =====================================================================================================================================
        clipLimit = 20  # -1
        tileGridSize = 8  # 1
        clahe = cv2.createCLAHE(clipLimit=clipLimit/10,
                                tileGridSize=(tileGridSize, tileGridSize))
        CLAHE = clahe.apply(FEN)

        # =====================================================================================================================================
        # https://www.twblogs.net/a/5b88eaed2b71775d1cdeeb3f
        # Rolling Guidance Filter
        # 在去移除和平滑圖像中的複雜的小區域時，還能保證大區域物體邊界的準確性。
        # 去除複雜背景，獲取物體輪廓，方便圖像分割。同時用其逆運算，可以增強圖像細節。
        # =====================================================================================================================================
        RGF = cv2.ximgproc.rollingGuidanceFilter(CLAHE, cv2.CV_32F)
        self.curr = RGF
        self.img = RGF


    # Mask
    def findMask(self):
        step = 2
        up = 10
        down = 1
        sato_image_Black = sato(
            self.curr, sigmas=range(9, 10, 1), black_ridges=True)

        sato_image = img_as_ubyte(sato_image_Black)

        img = self.curr
        img = img_as_float(img)
        # 黑白轉為RGB 1024*1024 => 1024*1024*3
        img = np.repeat(img, 3).reshape((img.shape[0], img.shape[1], -1))
        MASK = ((sato_image <= 68) & (sato_image >= 20))
        sato_image[MASK] = 255
        sato_image[~MASK] = 0

        # 骨架變細前
        kernel = np.ones((3, 3), np.uint8)
        sato_image = cv2.erode(sato_image, kernel, iterations=2)

        
        self.curr = sato_image
        self.img = img


        # 找到骨架
        sato_image = cv2.ximgproc.thinning(
            sato_image, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)

        # Connected Component
        ret, labels = cv2.connectedComponents(sato_image)
        for i in range(np.max(labels)):
            th = sum(np.where(labels.flatten() == i, 1, 0))
            if th < 100:
                sato_image = np.where(labels == i, 0, sato_image)

        MASK = sato_image == 255
        self.MASK = MASK
        # self.curr = img

        return MASK

    def ga_use_beforevote(self):
        voteAlgorithms = [
            QuickShiftVoter,
            FelzenszwalbVoter,
            SLICVoter,
            WatershedVoter
        ]

        img = self.img

        # ================================================================================================
        # 使用 Voter運算
        # ================================================================================================
        voters = []
        for va in voteAlgorithms:
            _ = va(img)
            _.process(self.MASK)
            voters.append(_)
            # _.save(ORIGINAL_IMG, "")

        Ground_Truth = self.labelImg
        Ground_Truth = np.where(Ground_Truth == 38, 255,
                                Ground_Truth)  # 38 是導管 轉換成 血管
        Ground_Truth = Ground_Truth.flatten()
        Ground_Truth = np.where(Ground_Truth == 255, 1,
                                Ground_Truth)  # 38 是導管 轉換成

        IMAGES = [
            v.GUSSEED_VESSEL_IMAGE.flatten() for v in voters
        ]

        for im in IMAGES:
            im = np.where(im == 255, 1, im)  # 38 是導管

        return {
            "filename": self.filename,
            "voters": IMAGES
        }, Ground_Truth

    # Vote

    def vote(self, weights):
        self.weights = weights
        voteAlgorithms = [
            QuickShiftVoter,
            FelzenszwalbVoter,
            SLICVoter,
            WatershedVoter
        ]

        # img = self.curr
        img = self.img

        # ================================================================================================
        # 使用 Voter運算
        # ================================================================================================
        voters = []
        for va in voteAlgorithms:
            _ = va(img)
            _.process(self.MASK)
            voters.append(_)
            # _.save(ORIGINAL_IMG, "")

        Ground_Truth = self.labelImg
        Ground_Truth = np.where(Ground_Truth == 38, 255,
                                Ground_Truth)  # 38 是導管 轉換成 血管
        Ground_Truth = Ground_Truth.flatten()
        Ground_Truth = np.where(Ground_Truth == 255, 1,
                                Ground_Truth)  # 38 是導管 轉換成

        IMAGES = [
            v.GUSSEED_VESSEL_IMAGE.flatten() for v in voters
        ]

        for im in IMAGES:
            im = np.where(im == 255, 1, im)  # 38 是導管

        standard = self.weights
        VOTE_THRESHOLD = sum(standard)/2
        VOTE_IMAGE = np.zeros(img.shape[:2])
        sum_weight_image = np.zeros(VOTE_IMAGE.shape)

        for a, b in zip(standard, voters[:4]):
            sum_weight_image = sum_weight_image + a * b.GUSSEED_VESSEL_IMAGE

        VOTE_IMAGE = np.where(sum_weight_image >= VOTE_THRESHOLD, 255, 0)
        VOTE_IMAGE = np.where(VOTE_IMAGE == 255, 1, 0)

        from pprint import pprint
        crD = classification_report(
            Ground_Truth[:], VOTE_IMAGE.flatten(), digits=5, output_dict=True)
        # print(self.filename, self.weights, crD["1"]["f1-score"])

        return crD["1"]["f1-score"]

    def refVote(self):
        voteAlgorithms = [
            QuickShiftVoter,
            FelzenszwalbVoter,
            SLICVoter,
            WatershedVoter
        ]

        img = self.curr

        # ================================================================================================
        # 使用 Voter運算
        # ================================================================================================
        voters = []
        for va in voteAlgorithms:
            _ = va(img)
            _.process(self.MASK)
            voters.append(_)
            # _.save(ORIGINAL_IMG, "")

        Ground_Truth = self.labelImg
        Ground_Truth = np.where(Ground_Truth == 38, 255,
                                Ground_Truth)  # 38 是導管 轉換成 血管
        Ground_Truth = Ground_Truth.flatten()
        Ground_Truth = np.where(Ground_Truth == 255, 1,
                                Ground_Truth)  # 38 是導管 轉換成

        IMAGES = [
            v.GUSSEED_VESSEL_IMAGE.flatten() for v in voters
        ]
        for im in IMAGES:
            im = np.where(im == 255, 1, im)

        crDs = []
        for im in IMAGES:
            crD = classification_report(
                Ground_Truth[:], im, digits=5, output_dict=True)
            crDs.append(
                crD["1"]["f1-score"]
            )

        return crDs

    # Upload

    def uploadResult(self, token):
        pass
