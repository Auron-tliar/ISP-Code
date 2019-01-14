import dlib
import os
from TrainingDataUtil import TrainingDataUtil

DATA_DIR = 'catfd\\data'
DETECTOR_SVM = os.path.join(DATA_DIR, 'detector.svm')
PREDICTOR_DAT = os.path.join(DATA_DIR, 'predictor.dat')


class Trainer:
    def __init__(self,
                 folder=TrainingDataUtil.training_data_dir,
                 cpu_cores=8,
                 window_size=200):
        self.folder = folder
        self.cpu_cores = cpu_cores
        self.xml = '{}/{}'.format(folder, TrainingDataUtil.training_data_xml)
        self.window_size = window_size

    def train_object_detector(self):
        self.__print_training_message('object detector')
        opt = dlib.simple_object_detector_training_options()
        opt.add_left_right_image_flips = True
        opt.C = 5
        opt.num_threads = self.cpu_cores
        opt.be_verbose = True
        opt.detection_window_size = self.window_size ** 2
        print(dlib.DLIB_USE_CUDA)
        dlib.train_simple_object_detector(self.xml, 'detector.svm', opt) #DETECTOR_SVM, opt)
        print("After train_simple_detector")

    def train_shape_predictor(self):
        self.__print_training_message('shape predictor')
        opt = dlib.shape_predictor_training_options()
        # opt.oversampling_amount = 300
        # opt.nu = 0.05
        # opt.tree_depth = 2
        opt.num_threads = self.cpu_cores
        opt.be_verbose = True
        print(dlib.DLIB_USE_CUDA)
        dlib.train_shape_predictor(self.xml, 'predictor.dat', opt) #PREDICTOR_DAT, opt)

    def view_object_detector(self):
        detector = dlib.simple_object_detector(DETECTOR_SVM)
        win_det = dlib.image_window()
        win_det.set_image(detector)
        dlib.hit_enter_to_continue()

    def __print_training_message(self, trainer):
        print 'Training {0} with {1} CPU cores.'.format(trainer, self.cpu_cores)
