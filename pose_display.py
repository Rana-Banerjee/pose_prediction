import numpy as np
import cv2
import io
import numpy as np
import openpifpaf
import PIL
from PIL import Image
import requests
import torch
#from painters import KeypointPainter
import matplotlib

openpifpaf.decoder.CifSeeds.threshold = 0.5
openpifpaf.decoder.nms.Keypoints.keypoint_threshold = 0.2
openpifpaf.decoder.nms.Keypoints.instance_threshold = 0.2
#RGB color pallate
COLORS = [
    (31,119,180),
    (174,199,232),
    (255, 127, 14),
    (255, 187, 120),
    (44, 160, 44),
    (152, 223, 138),
    (214, 39, 40),
    (255, 152, 150),
    (148, 103, 189),
    (197, 176, 213),
    (140, 86, 75),
    (196, 156, 148),
    (227, 119, 194),
    (247, 182, 210),
    (127, 127, 127),
    (199, 199, 199),
    (188, 189, 34),
    (219, 219, 141),
    (23, 190, 207),
    (158, 218, 229)
]

class PoseDisplay():
    def __init__(self):
        # Load a Trained Neural Network
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.net_cpu, _ = openpifpaf.network.factory(checkpoint='shufflenetv2k16w', download_progress=False)
        self.net = self.net_cpu.to(self.device)
        self.processor = openpifpaf.decoder.factory_decode(self.net.head_nets, basenet_stride=self.net.base_net.stride)


    # Function to perform pose prediction on given images
    def perform_pose_prediction(self,frame):

        # Load an Example Image
        #image_response = requests.get('https://raw.githubusercontent.com/vita-epfl/openpifpaf/master/docs/coco/000000081988.jpg')
        #pil_im = PIL.Image.open(frame).convert('RGB')
        #im = np.asarray(pil_im)
        im=frame
        #Comvert numpy array to PIL Image
        pil_im = Image.fromarray(im)

        #with openpifpaf.show.image_canvas(im) as ax:
        #    pass
    

        # Preprocessing, Dataset
        preprocess = openpifpaf.transforms.Compose([
            openpifpaf.transforms.NormalizeAnnotations(),
            openpifpaf.transforms.CenterPadTight(16),
            openpifpaf.transforms.EVAL_TRANSFORM,
            ])
        data = openpifpaf.datasets.PilImageList([pil_im], preprocess=preprocess)

        # DataLoader, Visualizer
        loader = torch.utils.data.DataLoader(
        data, batch_size=1, pin_memory=True, 
        collate_fn=openpifpaf.datasets.collate_images_anns_meta)

        # Prediction
        for images_batch, _, __ in loader:
            predictions = self.processor.batch(self.net, images_batch, device=self.device)[0]

        
        return predictions

    # Function to display webcam output with pose overlays
    def display_output(self):
        cap = cv2.VideoCapture(0)
        #keypoint_painter = KeypointPainter(color_connections=True, linewidth=6)
        count = 0 
        while(True):
            # Capture frame-by-frame
            ret, frame = cap.read()
            
            # Our operations on the frame come here
            overlay_output=frame

            # Call the perform_pose_prediction with the captured frame and get the resulting overlayed frame
            predictions = self.perform_pose_prediction(frame)
            for i, ann in enumerate(predictions):
                kps = ann.data
                assert kps.shape[1] == 3
                x = kps[:, 0] * 1
                y = kps[:, 1] * 1
                v = kps[:, 2]
                skeleton = ann.skeleton
                if not np.any(v > 0):
                    break
                lines, line_colors, line_styles = [], [], []
                for ci, (j1i, j2i) in enumerate(np.array(skeleton) - 1):
                    if v[j1i] > 0 and v[j2i] > 0:
                        lines.append([(x[j1i], y[j1i]), (x[j2i], y[j2i])])
                        line_colors.append(COLORS[ci][::-1] )
                        if v[j1i] > 0.5 and v[j2i] > 0.5:
                            line_styles.append('solid')
                        else:
                            line_styles.append('dashed')
                for i, (line,line_color) in enumerate(zip(lines,line_colors)):
                    start_point = line[0]
                    end_point = line[1]
                    overlay_output = cv2.line(overlay_output, start_point, end_point,line_color , 2)
                    #print(i, lines, line_colors, line_styles)


            # Display the resulting frame
            cv2.imshow('Figure',overlay_output)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        # When everything done, release the capture
        cap.release()
        cv2.destroyAllWindows()



pd = PoseDisplay()
pd.display_output()