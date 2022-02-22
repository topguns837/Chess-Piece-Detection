
# import the opencv library
import cv2
from tensorflow.keras.models import load_model
import tensorflow as tf
import numpy as np

def load_pb(path_to_pb):
    with tf.gfile.GFile(path_to_pb, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name='')
        return graph


# Load the model

model_path = 'C:/Users/Asus/Desktop/Chess_Piece_Images/Models/Validation_75%/sample_model.h5'
#model = tf.saved_model.load(model_path)
model = tf.keras.models.load_model(model_path)

IMG_SIZE = 200  
CATEGORIES = [ "Black_Bishop" , "White_Bishop" , "Black_Knight" , "White_Knight" , "Black_Rook" , "White_Rook" , 
              "Black_King" , "White_King"  , "Black_Queen" , "White_Queen" , "Black_Pawn" , "White_Pawn"]
  
# define a video capture object
vid = cv2.VideoCapture(1)
  
while(True):
      
    # Capture the video frame
    # by frame
    ret, frame = vid.read()
  
    # Display the resulting frame
    cv2.imshow('frame', frame)

    X= cv2.resize(frame , (IMG_SIZE , IMG_SIZE))
    X = np.array(X).reshape(-1, IMG_SIZE , IMG_SIZE , 3) 
    
    img_array = X/255.0

    X = tf.constant(img_array , dtype = tf.float64)
    y_prob = model.predict(X)

    y = tf.argmax(y_prob , axis = 1)
    y_class = CATEGORIES[int(y)]

    print(y_class)

      
    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  
# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()