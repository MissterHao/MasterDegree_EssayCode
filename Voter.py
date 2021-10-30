import abc
import matplotlib.pyplot as plt
import numpy as np

from skimage.color import rgb2gray
from skimage.filters import sobel
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from skimage.segmentation import mark_boundaries


QuickShift_DIRPATH   = "./imgdata/QuickShift/"
Felzenszwalb_DIRPATH   = "./imgdata/Felzenszwalb/"
SLIC_DIRPATH   = "./imgdata/SLIC/"
Watershed_DIRPATH   = "./imgdata/Watershed/"


QuickShift_DIRPATH   = "./for_yzu_essay/"
Felzenszwalb_DIRPATH   = "./for_yzu_essay/"
SLIC_DIRPATH   = "./for_yzu_essay/"
Watershed_DIRPATH   = "./for_yzu_essay/"

class Voter(metaclass=abc.ABCMeta):

    def __init__(self):
        self.hadsetconfig = False

    @abc.abstractmethod
    def setConfig(self, **kwargs):
        return NotImplemented

    @abc.abstractmethod
    def process(self):
        return NotImplemented

    @abc.abstractmethod
    def __getitem__(self, pos):
        return NotImplemented


    def __add__(self, anotherVoter):
        return self.GUSSEED_VESSEL_IMAGE + anotherVoter.GUSSEED_VESSEL_IMAGE



class QuickShiftVoter(Voter):

    def __init__(self, image: np.ndarray):
        super().__init__()
        self.image = image

    def setConfig(self, **kwargs):
        pass

    def process(self, MASK):
        # 設定參數
        self.max_dist_ = 3
        #
        self.original_segments_quick = quickshift(
            self.image, kernel_size=5, max_dist=self.max_dist_, ratio=0.3)
        # print(f"Quickshift number of segments: {len(np.unique(segments_quick))}")
        # 使用 MASK 來選擇
        selected_region = self.original_segments_quick[MASK]

        unique, counts = np.unique(selected_region, return_counts=True)
        unique = unique[(counts >= np.percentile(counts, 75)) & (counts != 0)]

        self.segments_quick = np.where(
            np.isin(self.original_segments_quick, np.unique(selected_region)),
            self.original_segments_quick, 0
        )

        self.GUSSEED_VESSEL_IMAGE = np.where(self.segments_quick > 0, 1, 0)

    def show(self, ORIGINAL_IMG):
        markedImage = mark_boundaries(ORIGINAL_IMG, self.segments_quick)
        plt.imshow(markedImage)
        plt.title(f"quickshift max_dist {self.max_dist_}")
        plt.show()

    
    def __getitem__(self, pos):
        return self.GUSSEED_VESSEL_IMAGE[pos[0], pos[1]]

    
    def save(self, ORIGINAL_IMG, name):
        path = QuickShift_DIRPATH
        markedImage = mark_boundaries(ORIGINAL_IMG, self.segments_quick)
        plt.figure()
        plt.imshow(markedImage)
        plt.title(f"{path}{name} {self.__class__.__name__[:-5]}.png")
        plt.axis('off')
        plt.savefig(f"{path}{name} {self.__class__.__name__[:-5]}.png", dpi=300)

        
        # Original
        markedImage = mark_boundaries(ORIGINAL_IMG, self.original_segments_quick)
        plt.imshow(markedImage)
        plt.title(f"{path}{name} {self.__class__.__name__[:-5]}.png")
        plt.axis('off')
        plt.savefig(f"{path}{name} non-masked {self.__class__.__name__[:-5]}.png", dpi=300)



class FelzenszwalbVoter(Voter):

    def __init__(self, image: np.ndarray):
        super().__init__()
        self.image = image

    def setConfig(self, **kwargs):
        pass

    def process(self, MASK):
        # 設定參數
        # self.scale_ = 10
        # self.sigma_ = 0.5
        # self.min_size_ = 500
        self.scale_ = 100
        self.sigma_ = 0.5
        self.min_size_ = 50
        self.original_segments_fz = felzenszwalb(self.image, scale=self.scale_, sigma=self.sigma_, min_size=self.min_size_)


        # 使用 MASK 來選擇
        selected_region = self.original_segments_fz[MASK]

        unique, counts = np.unique(selected_region, return_counts=True)
        unique = unique[(counts >= np.percentile(counts, 75)) & (counts != 0)]

        self.segments_fz = np.where(
            np.isin(self.original_segments_fz, np.unique(selected_region)),
            self.original_segments_fz, 0
        )

        self.GUSSEED_VESSEL_IMAGE = np.where(self.segments_fz > 0, 1, 0)

        # print(f"Felzenszwalb number of segments: {len(np.unique(segments_fz))}")


    def show(self, ORIGINAL_IMG):
        markedImage = mark_boundaries(ORIGINAL_IMG, self.segments_fz)
        plt.imshow(markedImage)
        plt.title(f"Felzenszwalb scale: {self.scale_} sigma: {self.sigma_} min_size: {self.min_size_} ")
        plt.show()

    
    def __getitem__(self, pos):
        return self.GUSSEED_VESSEL_IMAGE[pos[0], pos[1]]

    
    def save(self, ORIGINAL_IMG, name):
        path = Felzenszwalb_DIRPATH
        markedImage = mark_boundaries(ORIGINAL_IMG, self.segments_fz)
        plt.imshow(markedImage)
        plt.title(f"{path}{name} {self.__class__.__name__[:-5]}.png")
        plt.axis('off')
        plt.savefig(f"{path}{name} {self.__class__.__name__[:-5]}.png", dpi=300)

        # Original
        markedImage = mark_boundaries(ORIGINAL_IMG, self.original_segments_fz)
        plt.imshow(markedImage)
        plt.title(f"{path}{name} {self.__class__.__name__[:-5]}.png")
        plt.axis('off')
        plt.savefig(f"{path}{name} non-masked {self.__class__.__name__[:-5]}.png", dpi=300)
        
        
        
class SLICVoter(Voter):

    def __init__(self, image: np.ndarray):
        super().__init__()
        self.image = image

    def setConfig(self, **kwargs):
        pass

    def process(self, MASK):

        self.n_segments_ = 1000
        self.compactness_ = 10.0
        self.sigma_ = 0.1
        self.min_size_factor_= 0.01
        self.max_size_factor_= 1

        # 設定參數
        self.original_segments_slic = slic(
            self.image, 
            n_segments=self.n_segments_, 
            compactness=self.compactness_, 
            sigma=self.sigma_,
            min_size_factor=self.min_size_factor_,
            max_size_factor=self.max_size_factor_
        )

        


        # 使用 MASK 來選擇
        selected_region = self.original_segments_slic[MASK]

        unique, counts = np.unique(selected_region, return_counts=True)
        unique = unique[(counts >= np.percentile(counts, 75)) & (counts != 0)]

        self.segments_slic = np.where(
            np.isin(self.original_segments_slic, np.unique(selected_region)),
            self.original_segments_slic, 0
        )

        self.GUSSEED_VESSEL_IMAGE = np.where(self.segments_slic > 0, 1, 0)

        # print(f"SLIC number of segments: {len(np.unique(segments_slic))}")


    def show(self, ORIGINAL_IMG):
        markedImage = mark_boundaries(ORIGINAL_IMG, self.segments_slic)
        plt.imshow(markedImage)
        plt.title(f"SLIC n_segments: {self.n_segments_} compactness: {self.compactness_} sigma: {self.sigma_} min_size_factor: {self.min_size_factor_} max_size_factor: {self.max_size_factor_}  ")
        plt.show()

    def __getitem__(self, pos):
        return self.GUSSEED_VESSEL_IMAGE[pos[0], pos[1]]  
        
        
    
    def save(self, ORIGINAL_IMG, name):
        path = SLIC_DIRPATH
        markedImage = mark_boundaries(ORIGINAL_IMG, self.segments_slic)
        plt.imshow(markedImage)
        plt.title(f"{path}{name} {self.__class__.__name__[:-5]}.png")
        plt.axis('off')
        plt.savefig(f"{path}{name} {self.__class__.__name__[:-5]}.png", dpi=300)

        
        # Original
        markedImage = mark_boundaries(ORIGINAL_IMG, self.original_segments_slic)
        plt.imshow(markedImage)
        plt.title(f"{path}{name} {self.__class__.__name__[:-5]}.png")
        plt.axis('off')
        plt.savefig(f"{path}{name} non-masked {self.__class__.__name__[:-5]}.png", dpi=300)



class WatershedVoter(Voter):

    def __init__(self, image: np.ndarray):
        super().__init__()
        self.image = image

    def setConfig(self, **kwargs):
        pass

    def process(self, MASK):

        # Proposal 用
        self.markers_ = 5000
        self.compactness_ = 100.0

        # self.markers_ = 10000
        # self.compactness_ = 0


        # 設定參數
        gradient = sobel(rgb2gray(self.image))
        self.original_segments_watershed = watershed(gradient, markers=self.markers_, compactness=self.compactness_)


        # WS
        # from skimage.feature import peak_local_max
        # from scipy import ndimage as ndi

        # distance = ndi.distance_transform_edt(self.image)
        # local_maxi = peak_local_max(distance, indices=False, footprint=np.ones((3, 3)),
        #                             labels=self.image)
        # markers = ndi.label(local_maxi)[0]
        # segments_watershed = watershed(-distance, markers)



        


        # 使用 MASK 來選擇
        selected_region = self.original_segments_watershed[MASK]

        unique, counts = np.unique(selected_region, return_counts=True)
        unique = unique[(counts >= np.percentile(counts, 75)) & (counts != 0)]

        self.segments_watershed = np.where(
            np.isin(self.original_segments_watershed, np.unique(selected_region)),
            self.original_segments_watershed, 0
        )

        self.GUSSEED_VESSEL_IMAGE = np.where(self.segments_watershed > 0, 1, 0)

        # print(f"Watershed number of segments: {len(np.unique(segments_watershed))}")


    def show(self, ORIGINAL_IMG):
        markedImage = mark_boundaries(ORIGINAL_IMG, self.segments_watershed)
        plt.imshow(markedImage)
        plt.title(f"Watershed markers_: {self.markers_} compactness_: {self.compactness_} ")
        plt.show()

    def __getitem__(self, pos):
        return self.GUSSEED_VESSEL_IMAGE[pos[0], pos[1]]  

    
    def save(self, ORIGINAL_IMG, name):
        path = Watershed_DIRPATH
        markedImage = mark_boundaries(ORIGINAL_IMG, self.segments_watershed)
        plt.imshow(markedImage)
        plt.title(f"{path}{name} {self.__class__.__name__[:-5]}.png")
        plt.axis('off')
        plt.savefig(f"{path}{name} {self.__class__.__name__[:-5]}.png", dpi=300)

        
        # Original
        markedImage = mark_boundaries(ORIGINAL_IMG, self.original_segments_watershed)
        plt.imshow(markedImage)
        plt.title(f"{path}{name} {self.__class__.__name__[:-5]}.png")
        plt.axis('off')
        plt.savefig(f"{path}{name} non-masked {self.__class__.__name__[:-5]}.png", dpi=300)