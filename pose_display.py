import numpy as np
import cv2
import io
import numpy as np
import openpifpaf
import PIL
from PIL import Image
import requests
import torch

openpifpaf.decoder.CifSeeds.threshold = 0.5
openpifpaf.decoder.nms.Keypoints.keypoint_threshold = 0.2
openpifpaf.decoder.nms.Keypoints.instance_threshold = 0.2

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
        keypoint_painter = openpifpaf.show.KeypointPainter(color_connections=True, linewidth=6)

        while(True):
            # Capture frame-by-frame
            ret, frame = cap.read()

            # Our operations on the frame come here
            overlay_output = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Call the perform_pose_prediction with the captured frame and get the resulting overlayed frame
            predictions = self.perform_pose_prediction(frame)
            
            #for i, ann in enumerate(predictions):
            #    print(i, ann)

            with openpifpaf.show.image_canvas(overlay_output) as ax:
                keypoint_painter.annotations(ax, predictions)
                break
            
            # Create a overlay of the predictions on the frame
            #overlay_output = frame 
            # Display the resulting frame
            #cv2.imshow('frame',overlay_output)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        # When everything done, release the capture
        cap.release()
        cv2.destroyAllWindows()


pd = PoseDisplay()
pd.display_output()