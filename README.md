the project is about the ID card recognize, trying to use traditional image process methods, no involved deep leaning methods, just try to find out a way to recognize the ID card contents.

Why only use traditional image process methods?

it is for the resource saving, if using deep learning, the GPU is necessary, you have to ask client to purchase GPU server which is really costly. By traditional image porocess methods, client can use his old assets without new investment.

This is a experimental project, and I will try variable methods as below:

- haars
- morphology
- hough 
- mser
- radon
- sift
- watershed
- face recognition
- grabcut
- ...(more)

and the final purpose is to find out a feasible method to implement the general ID card image recongnition.

you need to prepare you data in [data] directory.

and you can run each code in same mode,like:

`python card/hough.py data/1.jpg`, others is like so.

[@piginzoo](www.piginzoo.com)