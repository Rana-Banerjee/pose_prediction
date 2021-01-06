import numpy as np
import cv2

# Function to display webcam output with pose overlays
def display_output():
    cap = cv2.VideoCapture(0)

    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Our operations on the frame come here
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Call the perform_pose_prediction with the captured frame and get the resulting overlayed frame

        # Display the resulting frame
        cv2.imshow('frame',gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


import io
import numpy as np
import openpifpaf
import PIL
import requests
import torch
# Function to perform pose prediction on given images
def perform_pose_prediction():

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    print(openpifpaf.__version__)
    print(torch.__version__)

    # Load an Example Image
    image_response = requests.get('https://raw.githubusercontent.com/vita-epfl/openpifpaf/master/docs/coco/000000081988.jpg')
    pil_im = PIL.Image.open(io.BytesIO(image_response.content)).convert('RGB')
    im = np.asarray(pil_im)

    #with openpifpaf.show.image_canvas(im) as ax:
    #    pass
    # Load a Trained Neural Network
    net_cpu, _ = openpifpaf.network.factory(checkpoint='shufflenetv2k16w', download_progress=False)
    net = net_cpu.to(device)

    openpifpaf.decoder.CifSeeds.threshold = 0.5
    openpifpaf.decoder.nms.Keypoints.keypoint_threshold = 0.2
    openpifpaf.decoder.nms.Keypoints.instance_threshold = 0.2
    processor = openpifpaf.decoder.factory_decode(net.head_nets, basenet_stride=net.base_net.stride)

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

    keypoint_painter = openpifpaf.show.KeypointPainter(color_connections=True, linewidth=6)

    # Prediction
    for images_batch, _, __ in loader:
        predictions = processor.batch(net, images_batch, device=device)[0]
    with openpifpaf.show.image_canvas(im) as ax:
        keypoint_painter.annotations(ax, predictions)

perform_pose_prediction()