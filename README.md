# FSLAM

if you are using virtual environments such as parallels, you need to do the following steps before you run build.sh:

edit  Thirdparty/Pangolin/src/display/device/display_x11.cpp and change the following line:
        GLX_DOUBLEBUFFER    , glx_doublebuffer ? True : False,
to
        GLX_DOUBLEBUFFER    , glx_doublebuffer ? False : False,




