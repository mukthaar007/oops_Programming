import numpy as np
import cv2
def find_location(outputs):
    bounding_boxes=[]
    class_ids=[]
    confidence=[]


    for k in outputs:
        for l in k:
            print(l)
            class_prob=l[5:]
            class_prob_high=np.argmax(class_prob)
            confidence_value=class_prob[class_prob_high]

            if confidence_value>threshold:
                w,h=int(l[2]*image_size),int(l[3]*image_size)
                x,y=int(l[0]*image_size-w/2),int(l[1]*image_size-h/2)
                bounding_boxes.append([x,y,w,h])
                class_ids.append(class_prob_high)
                confidence.append(float(confidence_value))

        indeces=cv2.dnn.NMSBoxes(bounding_boxes,confidence,threshold,0.5)
        return indeces, bounding_boxes, confidence, class_ids








#Reading image
#image=cv2.vidio('car.jpg')

#original_height, original_width=image.shape[:2]
#total 80 class files and load it

class_names=[]
k=open('classes','r')
for i in k.readlines():
    class_names.append(i.strip())
print(class_names)
#Load the model
Neural_network=cv2.dnn.readNetFromDarknet('yolov3.cfg','yolov3.weights')
cap=cv2.VideoCapture("test.mp4")
while cap.read():
    res,frame=cap.read()
    if res==True:
        original_height, original_width = frame.shape[1],frame.shape[0]
        blob = cv2.dnn.blobFromImage(frame, 1 / 255, (320, 320), True, crop=False)



#resize the input image shape and make them how model accepts

#Taking the 6300 boxes and selecting best
    threshold=0.3
    image_size=320



    #blob=cv2.dnn.blobFromImage(image,1/255,(320,320),True, crop=False)
    #print(blob.shape)

    #Getting layer names
    layers=Neural_network.getLayerNames()
    #print(layers)

    #Get outcome layers
    outcome_layers=Neural_network.getUnconnectedOutLayersNames()
    #print(outcome_layers)

    #Finding the index manually
    layer_index=Neural_network.getUnconnectedOutLayers()
    #print(layer_index)

    #Since index starts fro zero we need to reduce one value
    reduce_layer_index=[layers[j-1] for j in Neural_network.getUnconnectedOutLayers()]
    #print(reduce_layer_index)

    #Sending image to model
    Neural_network.setInput(blob)

    #Taking the outcomes from yolo82, yolo94, yolo106
    output=Neural_network.forward(reduce_layer_index)
    #print(output)
    #print(output[0].shape)
    prediction_box, bounding_box, cof, classes=find_location(output)
    print(prediction_box)
    print(bounding_box)
    print(cof)
    print(classes)



    font=cv2.FONT_HERSHEY_COMPLEX
    height_ratio=original_height/320
    width_ratio=original_width/320
    for pred in prediction_box.flatten():
        x,y,w,h=bounding_box[pred]
        x=int(x*width_ratio)
        y=int(y*height_ratio)
        w=int(w*width_ratio)
        h=int(h*height_ratio)
        label=str(class_names[classes[pred]])
        conf=cof[pred]
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
        cv2.putText(frame,label,(x,y-2),font,1,(0,0,0),1)


    cv2.imshow('image',frame)
    if cv2.waitKey(1) & 0xFF ==ord('q'):

        break
    else:
        break



cv2.destroyAllWindows()

