# deltaMush
This project is a maya deformer known as deltaMush, this deformer performs a regular average of the vertexs on bind pose, stores the amount of loss of volume per vertex and then at runtime perform the smoothing plus re-appling the delta in tangent space to compensate for the loss of volume.
I took my original version of the deformer and then performed a seried of optimizations in the CPU side of things, from getting better data locality and consecutive access acess of memory to exploit CPU memory caches to reversing where needed if statemnts to help instruction caches and branch prediciton aswell as adding multi-threading, all this can be found on master. 
Then I started working on a cuda and opencl implementation.
The original version was running on a rig at 17 fps with 20 smooth iterations, meanhwile the rig without deformer runs at 28.2~ FPS. With only CPU optimization I got the deformer up to 25 FPS, the cuda version allowed me to get close to the metal and hit 28 FPS, so only a 0.2 FPS drop which is really good. The openCL version instead was runnig much much more faster because was Maya pipeline friendly, the whole deformation pipe was on GPU , skincluster + deltaMush, and there is no copy memory back and forth neither conversion like in the CUDA version.

Recap of branches:
-Master : CPU version
-Cuda : Cuda version
-OpenCL : Opencl version

#TODO
Due to a bug in the Linux Nvidia driver (v352~, not sure yet if the v355~ fixes the problem ) which if you have a opencl deformer and you try to select the mesh you get a crash in the OpenGL, this bug has been flagged and aknowledged by Autodesk that was able to reproduce. That problems gives  hard time to manipulate attributes, for this reason the OpenCL version implements the basic version of the algorithm and doesnt allow attribute manipulations and I had not time to continue development under Windows. 
