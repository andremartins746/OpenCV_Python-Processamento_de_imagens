import numpy as np
import cv2

#caminhos dos videos
VIDEO_SOURSE = 'videos/Cars.mp4'
VIDEO_OUT = 'videos/results/filtragem_mediana_temporal.avi'

#amarzenando o video em uma variavel
cap = cv2.VideoCapture(VIDEO_SOURSE)
#lendo o video
hasFrame, frame = cap.read()
#exibindo as caracteristicas do video
#print(hasFrame, frame.shape)

#definindo a extenção do video e salvando ele
fourcc = cv2.VideoWriter_fourcc(* 'XVID')
writer = cv2.VideoWriter(VIDEO_OUT,fourcc, 25, (frame.shape[1], frame.shape[0]), False)

#mostrando quantos frames tem o video
#print(cap.get(cv2.CAP_PROP_FRAME_COUNT))

#gerando 25 numeros aleatorios uniformemente seguindo a Mediana
#print(np.random.uniform(size=25))

framesIds = cap.get(cv2.CAP_PROP_FRAME_COUNT) * np.random.uniform(size=25)
#print(framesIds)


#mostrando um frame dos 25 gerados aleatoriamente
#108, 2000
#hasFrame, frame = cap.read()
#cv2.imshow('teste', frame)
#cv2.waitKey(0)

frames = []

for fid in framesIds:
    cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
    hasFrame, frame = cap.read()
    frames.append(frame)

#print(np.asarray(frames).shape)
#print(frames[0])
#print(frames[1])

#for frame in frames:
    #cv2.imshow('frame', frame)
   # cv2.waitKey(0)

medianFrame = np.median(frames, axis=0).astype(dtype=np.uint8)

#print(frame[0])
#print(medianFrame)
#cv2.imshow("median frame",medianFrame)
#cv2.waitKey(0)

cv2.imwrite('model_median_frame.jpg', medianFrame)

cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
grayMedianFrame = cv2.cvtColor(medianFrame, cv2.COLOR_BGR2GRAY)
#cv2.imshow('gray',grayMedianFrame)
#cv2.waitKey(0)

while (True):
    hasFrame, frame = cap.read()

    if not hasFrame:
        print('Error')
        break

    frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    dframe = cv2.absdiff(frameGray, grayMedianFrame)


    #th, dframe = cv2.threshold(dframe, 70, 255, cv2.THRESH_BINARY)
    th, dframe = cv2.threshold(dframe, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    print(th)
    cv2.imshow('frame', dframe)
    writer.write(dframe)



    if(cv2.waitKey(1) & 0xFF == ord('q')):
        break

writer.release()
cap.release()
